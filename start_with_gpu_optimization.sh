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

# Apply genrl library patch for multi-GPU support
echo ""
echo "Checking and applying genrl library patches..."
python3 -c "
import os
import sys

# Check if patch is needed
patch_needed = False
grpo_file = '.venv/lib/python3.10/site-packages/genrl/trainer/grpo_trainer.py'

if os.path.exists(grpo_file):
    with open(grpo_file, 'r') as f:
        content = f.read()
        # Check if patch is already applied
        if 'Handle different reward tensor dimensions' not in content:
            patch_needed = True
            print('✓ Patch needed for genrl library')
        else:
            print('✓ Patch already applied')
else:
    print('✗ genrl library not found at expected location')
    sys.exit(0)

if patch_needed:
    print('Applying rewards dimension fix...')
    # Read the file
    with open(grpo_file, 'r') as f:
        lines = f.readlines()
    
    # Find and replace the problematic section
    new_lines = []
    i = 0
    while i < len(lines):
        if 'with torch.no_grad():' in lines[i] and i > 420 and i < 440:
            # Found the target section
            new_lines.append(lines[i])
            # Skip the original problematic lines
            j = i + 1
            while j < len(lines) and 'advantages = torch.flatten' not in lines[j]:
                j += 1
            
            # Add the fixed code
            new_lines.append('            # Handle different reward tensor dimensions\\n')
            new_lines.append('            if rewards.dim() == 1:\\n')
            new_lines.append('                # 1D tensor case (single reward per sample)\\n')
            new_lines.append('                advantages = rewards - rewards.mean()\\n')
            new_lines.append('                if rewards.numel() > 1:\\n')
            new_lines.append('                    advantages /= rewards.std() + 1e-8\\n')
            new_lines.append('            else:\\n')
            new_lines.append('                # 2D tensor case (multiple rewards per sample)\\n')
            new_lines.append('                advantages = rewards - rewards.mean(dim=1, keepdim=True)\\n')
            new_lines.append('                if rewards.shape[1] > 1:\\n')
            new_lines.append('                    advantages /= rewards.std(dim=1, keepdim=True) + 1e-8\\n')
            
            # Continue from where we left off
            i = j
        else:
            new_lines.append(lines[i])
            i += 1
    
    # Backup original file
    import shutil
    backup_file = grpo_file + '.backup'
    if not os.path.exists(backup_file):
        shutil.copy(grpo_file, backup_file)
        print(f'✓ Backup created: {backup_file}')
    
    # Write the patched file
    with open(grpo_file, 'w') as f:
        f.writelines(new_lines)
    
    print('✓ Patch applied successfully')
"

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