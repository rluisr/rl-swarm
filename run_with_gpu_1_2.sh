#!/bin/bash
# Script to run RL Swarm using only GPUs 1 and 2

echo "======================================"
echo "RL Swarm - Using GPUs 1 and 2 Only"
echo "======================================"

# Set environment variables for GPU selection
export CUDA_VISIBLE_DEVICES=1,2
echo "✓ CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
echo "✓ PyTorch memory optimization enabled"

# Clear GPU cache before starting
echo ""
echo "Clearing GPU cache..."
python3 -c "
import torch
if torch.cuda.is_available():
    # When CUDA_VISIBLE_DEVICES=1,2, PyTorch sees them as devices 0,1
    print(f'PyTorch sees {torch.cuda.device_count()} GPU(s)')
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
            props = torch.cuda.get_device_properties(i)
            print(f'  Device {i}: {props.name} ({props.total_memory / (1024**3):.2f} GB)')
    torch.cuda.synchronize()
    print('✓ GPU cache cleared')
"

# Check available memory
echo ""
echo "Current GPU status (physical GPUs 1 and 2):"
nvidia-smi --id=1,2 --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv

# Run the training
echo ""
echo "Starting training with GPUs 1 and 2..."
echo "======================================"
./start_with_gpu_optimization.sh "$@"