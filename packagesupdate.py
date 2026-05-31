# ==============================================================================
# CELL 1: THE NUKE & PAVE INSTALLER (UPDATED FOR PYTORCH 2.9)
# This forcefully uninstalls broken/pre-loaded versions and installs the correct ones.
# ==============================================================================

print("🧹 1. Forcefully uninstalling old and conflicting packages...")
# Added torchvision and torchaudio to the nuke list to prevent dependency resolver crashes
!pip uninstall -y -q torch torch_xla torchvision torchaudio transformers datasets trl accelerate huggingface_hub einops tiktoken

print("📥 2. Installing PyTorch 2.9.0 and the matching Google TPU XLA drivers...")
# UPDATED: Explicitly pinned to 2.9.0 to match the available Colab TPU releases
!pip install -q torch==2.9.0 torch_xla[tpu]==2.9.0 -f https://storage.googleapis.com/libtpu-releases/index.html

print("🤗 3. Installing the latest Hugging Face ecosystem...")
!pip install -q -U transformers datasets trl accelerate huggingface_hub einops tiktoken

print("\n====================================================================")
print("✅ INSTALLATION COMPLETE!")
print("🛑 STOP! YOU MUST RESTART THE SESSION NOW.")
print("Go to the top menu: Runtime -> Restart Session")
print("Do NOT run Cell 2 until you have done this!")
print("====================================================================")
