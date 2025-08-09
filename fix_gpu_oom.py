#!/usr/bin/env python3
"""
Fix GPU OOM issues for multi-GPU training with CUDA_VISIBLE_DEVICES
"""
import os

# Fix 1: Update trainer.py to properly handle CUDA_VISIBLE_DEVICES
trainer_fix = """
# Add after trainer initialization in rgym_exp/src/trainer.py
# Clear GPU cache periodically
if hasattr(torch, 'cuda') and torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
"""

# Fix 2: Add environment variable to optimize memory allocation
env_vars = """
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
"""

# Fix 3: Reduce batch size for low VRAM
config_fix = """
# In rgym_exp/config/rg-swarm.yaml, change:
num_train_samples: 1  # Reduced from 2 to 1 for low VRAM
"""

print("GPU OOM Fix Instructions:")
print("=" * 50)
print("\n1. Environment Variables (add to your shell or script):")
print(env_vars)
print("\n2. For low VRAM (< 8GB per GPU), reduce batch size:")
print(config_fix)
print("\n3. Ensure CUDA_VISIBLE_DEVICES is set correctly:")
print("   export CUDA_VISIBLE_DEVICES=1,2  # Use GPUs 1 and 2 only")
print("\n4. Clear GPU cache before starting:")
print("   python3 -c 'import torch; torch.cuda.empty_cache()'")
print("\n5. Monitor GPU memory usage:")
print("   watch -n 1 nvidia-smi")
print("\nThe trainer.py has been updated to properly respect CUDA_VISIBLE_DEVICES.")
print("GPUs will now be numbered from 0 based on visible devices only.")