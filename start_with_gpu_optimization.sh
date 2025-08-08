#!/bin/bash

# Start script with GPU optimization for RL Swarm
# This script helps prevent CUDA OutOfMemory errors by configuring optimal settings

echo "======================================"
echo "RL Swarm GPU Optimized Startup Script"
echo "======================================"

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA is available')
    print(f'✓ Found {torch.cuda.device_count()} GPU(s)')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory / (1024**3):.2f} GB)')
else:
    print('✗ CUDA is not available')
"

# Check current GPU memory status
echo ""
echo "Current GPU memory status:"
python3 rgym_exp/src/utils/gpu_monitor.py

# Set environment variables for optimal GPU usage
echo ""
echo "Setting GPU optimization environment variables..."

# Enable memory growth for TensorFlow (if used)
export TF_FORCE_GPU_ALLOW_GROWTH=true

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Enable NCCL optimizations for multi-GPU
if [ $(python3 -c "import torch; print(torch.cuda.device_count())") -gt 1 ]; then
    echo "Multiple GPUs detected, enabling NCCL optimizations..."
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
fi

# Ask user about configuration preferences
echo ""
echo "Configuration options:"
echo "1. Use default settings (recommended)"
echo "2. Enable fp16 training (saves memory)"
echo "3. Adjust batch size"
echo "4. View current configuration"
read -p "Select option (1-4): " option

case $option in
    2)
        echo "Enabling fp16 training..."
        # Update config to enable fp16
        sed -i.bak 's/fp16: false/fp16: true/' rgym_exp/config/rg-swarm.yaml
        echo "✓ fp16 training enabled"
        ;;
    3)
        read -p "Enter batch size (current: 2): " batch_size
        if [[ $batch_size =~ ^[0-9]+$ ]]; then
            sed -i.bak "s/num_train_samples: 2/num_train_samples: $batch_size/" rgym_exp/config/rg-swarm.yaml
            echo "✓ Batch size set to $batch_size"
        else
            echo "Invalid batch size, using default"
        fi
        ;;
    4)
        echo ""
        echo "Current configuration:"
        grep -E "fp16:|num_train_samples:|gradient_checkpointing:|gradient_accumulation_steps:" rgym_exp/config/rg-swarm.yaml
        ;;
    *)
        echo "Using default settings"
        ;;
esac

# Clear GPU cache before starting
echo ""
echo "Clearing GPU cache..."
python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print('✓ GPU cache cleared')
"

# Start the training
echo ""
echo "Starting RL Swarm with GPU optimizations..."
echo "======================================"
echo ""

# Check if running in Docker or native
if [ -f /.dockerenv ]; then
    echo "Running in Docker container..."
    python -m hydra._internal.hydra_main \
        --config-path "/home/gensyn/rl_swarm/configs" \
        --config-name "rg-swarm.yaml"
else
    echo "Running natively..."
    # Run the main launch script
    ./run_rl_swarm.sh
fi