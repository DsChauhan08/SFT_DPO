import os
import torch
import time
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from transformers.trainer_utils import get_last_checkpoint
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, HfApi

# ---------------------------------------------------------------------------
# 1. KAGGLE SECRETS & HUGGING FACE LOGIN
# ---------------------------------------------------------------------------
print("Authenticating...")
try:
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(token=hf_token)
    print("✅ Successfully logged into Hugging Face.")
except Exception as e:
    print(f"❌ Failed to login. Please attach 'HF_TOKEN' to Kaggle Secrets! Error: {e}")
    exit(1)

# ---------------------------------------------------------------------------
# 2. TPU XLA SETUP
# ---------------------------------------------------------------------------
import torch_xla.core.xla_model as xm
device = xm.xla_device()
print(f"✅ Successfully initialized TPU device: {device}")

# ---------------------------------------------------------------------------
# 3. DATASET LOADING & PARSING (High Redundancy)
# ---------------------------------------------------------------------------
def load_and_prepare_data():
    print("Loading datasets...")
    # Using 1k slices for secondaries to ensure rapid epoch completion and memory safety
    d_main = load_dataset("dschauhan08/gen-v1-dataset", split="train")
    d1 = load_dataset("allenai/tulu-3-sft-personas-instruction-following", split="train[:1000]")
    d2 = load_dataset("NousResearch/hermes-function-calling-v1", split="train[:1000]")
    d3 = load_dataset("qwedsacf/competition_math", split="train[:1000]")
    d4 = load_dataset("NousResearch/terminal-bench-2", split="train[:1000]")
    d5 = load_dataset("amphora/ResearchMath-14k", split="train[:1000]")
    
    def universal_standardize(example):
        if "messages" in example and isinstance(example["messages"], list):
            return {"messages": [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in example["messages"]]}
        if "conversations" in example and isinstance(example["conversations"], list):
            return {"messages": [{"role": "user" if t.get("from") in ["human", "user"] else "assistant", "content": t.get("value", "")} for t in example["conversations"]]}
        p = example.get("instruction") or example.get("prompt") or example.get("question")
        r = example.get("output") or example.get("response") or example.get("answer") or example.get("completion")
        if p and r:
            return {"messages": [{"role": "user", "content": str(p)}, {"role": "assistant", "content": str(r)}]}
        return {"messages": []}

    print("Formatting datasets...")
    all_d = [d_main.filter(lambda x: isinstance(x.get("messages"), list) and len(x["messages"]) > 0).remove_columns([c for c in d_main.column_names if c != "messages"])]
    for d in [d1, d2, d3, d4, d5]:
        std = d.map(universal_standardize, remove_columns=d.column_names).filter(lambda x: len(x.get("messages", [])) > 0)
        all_d.append(std)
        
    merged = concatenate_datasets(all_d).shuffle(seed=42)
    print(f"✅ Dataset ready. Total rows: {len(merged)}")
    return merged

dataset = load_and_prepare_data()

# ---------------------------------------------------------------------------
# 4. MODEL & TOKENIZER CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_ID = "openbmb/MiniCPM-1B-sft-bf16" # Specific chat-tuned base

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_chat_template(row):
    try:
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        # FORCE the model to stop by explicitly appending EOS token
        if not text.endswith(tokenizer.eos_token):
            text += tokenizer.eos_token
        return {"text": text}
    except:
        return {"text": ""}

dataset = dataset.map(format_chat_template).filter(lambda x: len(x["text"]) > 0)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# ---------------------------------------------------------------------------
# 5. TPU TRAINING ARGUMENTS (Mathematically secured for 16k Context)
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="/kaggle/working/sft-minicpm",
    per_device_train_batch_size=1,   # MANDATORY: 1 is the only way 16k context fits in 16GB
    gradient_accumulation_steps=16,  # Compensate to maintain Global Batch Size
    learning_rate=2e-5,
    num_train_epochs=3,              # Train for 3 SFT epochs
    bf16=True,                       # Hardware acceleration
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,              # Prevents Kaggle disk space crash
    optim="adafactor",               # Massive memory saver vs AdamW
    gradient_checkpointing=True,     # MANDATORY for 16k context
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",       
    max_seq_length=16384,            # Pushed to 16k context length!
    tokenizer=tokenizer,
    args=training_args,
)

# ---------------------------------------------------------------------------
# 6. TRAINING EXECUTION & AUTO-RESUME
# ---------------------------------------------------------------------------
last_checkpoint = get_last_checkpoint("/kaggle/working/sft-minicpm")
if last_checkpoint is not None:
    print(f"🔄 Resuming from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("🚀 Starting 3-Epoch SFT from scratch...")
    trainer.train() # YES, this triggers the training process.

# ---------------------------------------------------------------------------
# 7. AUTOMATIC PUSH TO HUGGING FACE
# ---------------------------------------------------------------------------
print("🎉 SFT Complete! Preparing to push to Hugging Face...")
FINAL_REPO = "dschauhan08/My-MiniCPM-Coder"
api = HfApi()

try:
    api.create_repo(repo_id=FINAL_REPO, exist_ok=True, token=hf_token)
    
    # Save the model locally first
    trainer.save_model("/kaggle/working/final-sft-model")
    tokenizer.save_pretrained("/kaggle/working/final-sft-model")
    
    # Push weights and tokenizer up to HF
    print(f"Uploading model directly to {FINAL_REPO}...")
    model.push_to_hub(FINAL_REPO, token=hf_token, safe_serialization=True)
    tokenizer.push_to_hub(FINAL_REPO, token=hf_token)
    print("✅ Model successfully uploaded to Hugging Face! Ready for RL/DPO.")
except Exception as e:
    print(f"❌ Upload failed, but model is saved locally at /kaggle/working/final-sft-model. Error: {e}")
