import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, HfApi

# ---------------------------------------------------------------------------
# 1. KAGGLE SECRETS
# ---------------------------------------------------------------------------
print("Authenticating...")
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)

import torch_xla.core.xla_model as xm
device = xm.xla_device()

# ---------------------------------------------------------------------------
# 2. LOAD SFT MODEL (The one you just trained and uploaded!)
# ---------------------------------------------------------------------------
MODEL_ID = "dschauhan08/My-MiniCPM-Coder" # Pulling your custom model

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True)
# DPO requires a reference model (the frozen original model) to compare against
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True)

# ---------------------------------------------------------------------------
# 3. PREFERENCE DATASET (Reinforcement Logic)
# ---------------------------------------------------------------------------
# DPO requires a prompt, a "chosen" (good) response, and a "rejected" (lazy) response.
# We use a high-quality math/coding preference dataset.
dataset = load_dataset("argilla/distilabel-math-preference-dpo", split="train[:5000]")

def format_dpo(example):
    # DPO trainer expects specific column names: prompt, chosen, rejected
    return {
        "prompt": example["instruction"],
        "chosen": example["chosen_response"] + tokenizer.eos_token,
        "rejected": example["rejected_response"] + tokenizer.eos_token
    }

dpo_dataset = dataset.map(format_dpo)

# ---------------------------------------------------------------------------
# 4. DPO TRAINING ARGUMENTS
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="/kaggle/working/dpo-minicpm",
    per_device_train_batch_size=1,   # Keeping it at 1 to prevent OOM
    gradient_accumulation_steps=8, 
    learning_rate=5e-6,              # RL requires a MUCH smaller learning rate than SFT
    num_train_epochs=1,              # 1 Epoch is usually enough for DPO
    bf16=True,
    logging_steps=10,
    optim="adafactor",
    gradient_checkpointing=True,
    report_to="none"
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    beta=0.1,                        # How heavily to penalize the model for diverging from the reference
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    max_prompt_length=4096,          
    max_length=8192                  # Reduced slightly from 16k because DPO loads TWO models into memory!
)

print("🚀 Starting Reinforcement Learning (DPO)...")
dpo_trainer.train()

# ---------------------------------------------------------------------------
# 5. PUSH FINALIZED RL MODEL TO HUB
# ---------------------------------------------------------------------------
FINAL_REPO = "dschauhan08/My-MiniCPM-Coder-RL"
print(f"🎉 RL Complete! Pushing ultimate model to {FINAL_REPO}...")
try:
    api = HfApi()
    api.create_repo(repo_id=FINAL_REPO, exist_ok=True, token=hf_token)
    model.push_to_hub(FINAL_REPO, token=hf_token, safe_serialization=True)
    tokenizer.push_to_hub(FINAL_REPO, token=hf_token)
    print("✅ Final RL Model uploaded! You are completely done.")
except Exception as e:
    print(f"❌ Upload failed. Model saved locally at /kaggle/working/dpo-minicpm. Error: {e}")
