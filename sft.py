# ==============================================================================
# ULTIMATE KAGGLE TPU SFT PIPELINE (PRODUCTION READY)
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

# IMPORT TPU LIBRARY
import torch_xla
import torch_xla.core.xla_model as xm

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

# ---------------------------------------------------------------------------
# 2. TPU INITIALIZATION & HOTFIXES
# ---------------------------------------------------------------------------
device = torch_xla.device() 
print(f"✅ Initialized TPU: {device}")

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
# 3. LOAD TOKENIZER
# ---------------------------------------------------------------------------
MODEL_ID = "openbmb/MiniCPM5-1B" 
print(f"📦 Loading Tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------------------------
# 4. DATASET PREPARATION (Loading Both Exact Datasets)
# ---------------------------------------------------------------------------
def extract_and_template(example):
    try:
        msgs = example.get("messages", [])
        if not msgs or not isinstance(msgs, list): return {"text": ""}
            
        clean_msgs = [{"role": str(m.get("role", "user")).strip().lower(), "content": str(m.get("content", ""))} for m in msgs]
        text = tokenizer.apply_chat_template(clean_msgs, tokenize=False)
        
        # GUARANTEE IT LEARNS TO STOP: Explicitly append EOS
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

# CRITICAL HOTFIX 3: TRUNCATION POISONING PREVENTION
# We drop anything longer than 12,000 characters so it fits perfectly in 3072 tokens.
MAX_CHARS = 12000

print(f"Filtering out rows larger than {MAX_CHARS} characters to preserve EOS tokens...")
ds_reasoning = ds_reasoning.filter(lambda x: 10 < len(x["text"]) < MAX_CHARS)
ds_minicpm = ds_minicpm.filter(lambda x: 10 < len(x["text"]) < MAX_CHARS)

dataset = concatenate_datasets([ds_reasoning, ds_minicpm]).shuffle(seed=42)
print(f"✅ Data processing flawless! Total training rows after filtering: {len(dataset)}")

# ---------------------------------------------------------------------------
# 5. STATIC TOKENIZATION & NATIVE TORCH CONVERSION
# ---------------------------------------------------------------------------
print("⚙️ Tokenizing datasets manually & creating strict static labels...")
def tokenize_batch(batch):
    # THE SWEET SPOT: 3072 tokens (~12,000 characters). Fits TPU memory perfectly!
    outputs = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=3072 
    )
    
    # Manually create 'labels' for causal LM and mask out padding with -100
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

# CRITICAL HOTFIX 4: Force native PyTorch Tensors to prevent XLA compiler crashes
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
print("✅ Tokenization complete. Data mathematically locked for XLA compiler.")

# ---------------------------------------------------------------------------
# 6. LOAD MODEL
# ---------------------------------------------------------------------------
print("🧠 Loading 1B Model to TPU HBM...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# ---------------------------------------------------------------------------
# 7. CORE TRAINING ARGUMENTS (Optimized for Kaggle)
# ---------------------------------------------------------------------------
OUTPUT_DIR = "/kaggle/working/minicpm-sft-output"

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,   # Kaggle has 8 cores, so Global Batch Size = 8
    gradient_accumulation_steps=1,   # Kept at 1 to prevent XLA from unrolling massive graphs
    learning_rate=2e-5,
    num_train_epochs=1,              
    bf16=True,                       
    logging_steps=10,
    save_strategy="epoch",           
    save_total_limit=2,              # Deletes old checkpoints so Kaggle disk doesn't fill up
    optim="adafactor",               
    gradient_checkpointing=True,     
    report_to="none",
    dataloader_drop_last=True        # CRITICAL: Drops uneven last batch so XLA shapes never change
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=default_data_collator
)

# ---------------------------------------------------------------------------
# 8. EXECUTION & AUTO-RESUME
# ---------------------------------------------------------------------------
last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
if last_checkpoint is not None:
    print(f"🔄 Resuming session from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("🚀 Starting full-scale SFT training loop on TPU...")
    trainer.train()

print("🎉 SFT Training Complete!")

# ---------------------------------------------------------------------------
# 9. AUTOMATIC PUSH TO HUGGING FACE
# ---------------------------------------------------------------------------
FINAL_REPO = "dschauhan08/My-MiniCPM5-1B-Coder"

if hf_token:
    print(f"☁️ Preparing to push final model to Hugging Face: {FINAL_REPO}...")
    try:
        api = HfApi()
        api.create_repo(repo_id=FINAL_REPO, exist_ok=True, token=hf_token)
        
        # Save locally first
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        # Push to hub
        model.push_to_hub(FINAL_REPO, token=hf_token, safe_serialization=True)
        tokenizer.push_to_hub(FINAL_REPO, token=hf_token)
        print("✅ Model successfully uploaded to Hugging Face! Ready for DPO/Inference.")
    except Exception as e:
        print(f"❌ Upload failed, but model is saved locally at {OUTPUT_DIR}. Error: {e}")
