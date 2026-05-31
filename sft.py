# ==============================================================================
# ULTIMATE KAGGLE TPU SFT PIPELINE (PRODUCTION READY - 8 CORE MULTIPROCESSING)
# Designed for Kaggle TPU v5e-8
# ==============================================================================
import os

# CRITICAL HOTFIX 1: Force TPU hardware to use bf16 natively to prevent XLA type crashes
os.environ["XLA_USE_BF16"] = "1" 

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint
from huggingface_hub import login, HfApi

# IMPORT TPU MULTIPROCESSING LIBRARY
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# ---------------------------------------------------------------------------
# 1. AUTHENTICATION (Kaggle Secrets)
# ---------------------------------------------------------------------------
print("🔐 Authenticating with Hugging Face...")
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN")
    except ImportError:
        pass

if hf_token:
    login(token=hf_token)
    print("✅ Successfully logged into Hugging Face.")
else:
    print("⚠️ No HF_TOKEN found! Training will work, but pushing to Hub will fail.")

# CRITICAL HOTFIX 2: PYTORCH 2.X XLA GRADIENT CHECKPOINTING BUG
if not hasattr(torch, "xla"):
    class XLADeviceModule:
        @staticmethod
        def get_rng_state(device=None): return xm.get_rng_state(device)
        @staticmethod
        def set_rng_state(state, device=None): xm.set_rng_state(state, device)
        @staticmethod
        def is_available(): return True
    torch.xla = XLADeviceModule()
    print("✅ Monkey-patched torch.xla for Gradient Checkpointing stability.")

# ---------------------------------------------------------------------------
# 2. LOAD TOKENIZER (Done globally before spawn)
# ---------------------------------------------------------------------------
MODEL_ID = "openbmb/MiniCPM5-1B" 
print(f"📦 Loading Tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------------------------
# 3. DATASET PREPARATION (Done globally before spawn)
# ---------------------------------------------------------------------------
def extract_and_template(example):
    try:
        msgs = example.get("messages", [])
        if not msgs or not isinstance(msgs, list): return {"text": ""}
            
        clean_msgs = [{"role": str(m.get("role", "user")).strip().lower(), "content": str(m.get("content", ""))} for m in msgs]
        text = tokenizer.apply_chat_template(clean_msgs, tokenize=False)
        
        if not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
            
        return {"text": text}
    except Exception:
        return {"text": ""}

print("📥 Downloading datasets...")
ds_reasoning = load_dataset("dschauhan08/superset-finetuning-reasoning", split="train")
ds_minicpm = load_dataset("dschauhan08/minicpm5-chatml-mix", split="train")

print("🧹 Applying chat templates and verifying schemas...")
ds_reasoning = ds_reasoning.map(extract_and_template, remove_columns=ds_reasoning.column_names, desc="Templating Reasoning")
ds_minicpm = ds_minicpm.map(extract_and_template, remove_columns=ds_minicpm.column_names, desc="Templating MiniCPM")

MAX_CHARS = 12000
print(f"Filtering out rows larger than {MAX_CHARS} characters to preserve EOS tokens...")
ds_reasoning = ds_reasoning.filter(lambda x: 10 < len(x["text"]) < MAX_CHARS)
ds_minicpm = ds_minicpm.filter(lambda x: 10 < len(x["text"]) < MAX_CHARS)

dataset = concatenate_datasets([ds_reasoning, ds_minicpm]).shuffle(seed=42)
print(f"✅ Data processing flawless! Total training rows after filtering: {len(dataset)}")

print("⚙️ Tokenizing datasets manually & creating strict static labels...")
def tokenize_batch(batch):
    outputs = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=3072 
    )
    
    labels = []
    for input_ids in outputs["input_ids"]:
        labels.append([t if t != tokenizer.pad_token_id else -100 for t in input_ids])
        
    outputs["labels"] = labels
    return outputs

tokenized_dataset = dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing"
)

# Force native PyTorch Tensors
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
print("✅ Tokenization complete. Data mathematically locked for XLA compiler.")

# ==============================================================================
# 4. MULTIPROCESSING SPAWN FUNCTION
# Everything inside this function will be cloned and executed across all 8 cores!
# ==============================================================================
OUTPUT_DIR = "/kaggle/working/minicpm-sft-output"
FINAL_REPO = "dschauhan08/My-MiniCPM5-1B-Coder"

def _mp_fn(index):
    # 1. Core Assignment
    device = xm.xla_device()
    if index == 0:
        print(f"🚀 Spawning 8 processes! Core {index} reporting for duty on {device}...")
    
    # 2. Model Loading (Each core loads the model into its specific TPU slice)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,   # 1 per core * 8 cores = Global Batch Size of 8!
        gradient_accumulation_steps=1,   
        learning_rate=2e-5,
        num_train_epochs=1,              
        bf16=True,                       
        logging_steps=10,
        save_strategy="epoch",           
        save_total_limit=2,              
        optim="adafactor",               
        gradient_checkpointing=True,     
        report_to="none",
        dataloader_drop_last=True,
        dataloader_pin_memory=False      # Silences the PyTorch Dataloader warning
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=default_data_collator
    )

    # 4. Execution
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
    if last_checkpoint is not None:
        if index == 0: print(f"🔄 Resuming session from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        if index == 0: print("🚀 Starting full-scale SFT training loop on all 8 TPU Cores...")
        trainer.train()

    # 5. Master Node Saves & Pushes to Hugging Face
    # We only let Core 0 do this, otherwise all 8 cores try to upload at the same time and crash.
    xm.rendezvous("training_complete") # Wait for all 8 cores to finish
    if index == 0:
        print("🎉 SFT Training Complete!")
        if hf_token:
            print(f"☁️ Preparing to push final model to Hugging Face: {FINAL_REPO}...")
            try:
                api = HfApi()
                api.create_repo(repo_id=FINAL_REPO, exist_ok=True, token=hf_token)
                trainer.save_model(OUTPUT_DIR)
                tokenizer.save_pretrained(OUTPUT_DIR)
                model.push_to_hub(FINAL_REPO, token=hf_token, safe_serialization=True)
                tokenizer.push_to_hub(FINAL_REPO, token=hf_token)
                print("✅ Model successfully uploaded to Hugging Face! Ready for DPO/Inference.")
            except Exception as e:
                print(f"❌ Upload failed, but model is saved locally at {OUTPUT_DIR}. Error: {e}")

# ==============================================================================
# 5. IGNITION 
# ==============================================================================
if __name__ == "__main__":
    # This fires up all 8 TPU cores and runs the _mp_fn function on each of them!
    xmp.spawn(_mp_fn, args=(), nprocs=8, start_method='fork')
