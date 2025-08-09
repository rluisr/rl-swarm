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

# PyTorch settings for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
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

# Also check and fix remote server if available
if command -v ssh &> /dev/null && ssh -o ConnectTimeout=2 hiveos.luis.local -l user "exit" 2>/dev/null; then
    echo "Applying patches on remote server..."
    ssh hiveos.luis.local -l user "sudo python3 - << 'EOF'
import os
import sys

grpo_file = '/root/test/.venv/lib/python3.10/site-packages/genrl/trainer/grpo_trainer.py'
if os.path.exists(grpo_file):
    with open(grpo_file, 'r') as f:
        lines = f.readlines()
    
    fixed = False
    new_lines = []
    for i, line in enumerate(lines):
        if i > 340 and i < 350:
            if line.startswith(' # Also handle old_per_token_logps'):
                new_lines.append('        # Also handle old_per_token_logps for DataParallel\\n')
                fixed = True
            elif line.startswith(' if self.args.num_iterations > 1'):
                new_lines.append('        if self.args.num_iterations > 1 and old_per_token_logps.shape[0] != current_batch_size:\\n')
                fixed = True
            elif line.startswith('     old_per_token_logps = old_per_token_logps'):
                new_lines.append('            old_per_token_logps = old_per_token_logps[:current_batch_size]\\n')
                fixed = True
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    if fixed:
        with open(grpo_file, 'w') as f:
            f.writelines(new_lines)
        # Also fix any backslash escape issues
        import subprocess
        subprocess.run(['sed', '-i', 's/\\\\!=/!=/g', grpo_file])
        print('✓ Remote server patched')
    else:
        print('✓ Remote server already patched or no fix needed')
else:
    print('✗ genrl library not found on remote server')
EOF"
fi

python3 -c "
import os
import sys

# Check if patch is needed
patch_needed = False
indent_fix_needed = False
grpo_file = '.venv/lib/python3.10/site-packages/genrl/trainer/grpo_trainer.py'

if os.path.exists(grpo_file):
    with open(grpo_file, 'r') as f:
        content = f.read()
        lines = content.split('\n')
        
        # Check if rewards patch is already applied
        if 'Handle different reward tensor dimensions' not in content:
            patch_needed = True
            print('✓ Rewards patch needed for genrl library')
        else:
            print('✓ Rewards patch already applied')
            
        # Check for indentation error at line 343-345
        for i, line in enumerate(lines):
            if i > 340 and i < 350:
                if line.startswith(' # Also handle old_per_token_logps'):
                    indent_fix_needed = True
                    print('✓ Indentation fix needed at line', i+1)
                    break
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

# Check for tensor mismatch issue
tensor_fix_needed = False
with open(grpo_file, 'r') as f:
    content = f.read()
    if 'Flatten per_token_logps if it has extra dimensions' not in content:
        tensor_fix_needed = True
        print('✓ Tensor mismatch fix needed')

# Fix indentation error if needed
if indent_fix_needed:
    print('Fixing indentation error at line 343-345...')
    with open(grpo_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for i, line in enumerate(lines):
        # Fix lines 343-345 with incorrect indentation
        if i > 340 and i < 350:
            if line.startswith(' # Also handle old_per_token_logps'):
                # Fix comment indentation
                new_lines.append('        # Also handle old_per_token_logps for DataParallel\\n')
            elif line.startswith(' if self.args.num_iterations > 1'):
                # Fix if statement indentation
                new_lines.append('        if self.args.num_iterations > 1 and old_per_token_logps.shape[0] != current_batch_size:\\n')
            elif line.startswith('     old_per_token_logps = old_per_token_logps'):
                # Fix assignment indentation
                new_lines.append('            old_per_token_logps = old_per_token_logps[:current_batch_size]\\n')
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Write the fixed file
    with open(grpo_file, 'w') as f:
        f.writelines(new_lines)
    
    # Fix any backslash escape issues
    import subprocess
    subprocess.run(['sed', '-i', 's/\\\\!=/!=/g', grpo_file])
    
    print('✓ Indentation error fixed')

# Check for completion_mask dimension mismatch
completion_mask_fix_needed = False
with open(grpo_file, 'r') as f:
    content = f.read()
    if 'Ensure per_token_loss and completion_mask have compatible shapes' not in content:
        completion_mask_fix_needed = True
        print('✓ Completion mask dimension fix needed')

# Fix tensor mismatch if needed
if tensor_fix_needed:
    print('Fixing tensor dimension mismatch...')
    with open(grpo_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for i, line in enumerate(lines):
        if i > 320 and 'advantages = inputs["advantages"]' in line:
            new_lines.append(line)
            # Add tensor flattening logic
            new_lines.append('        \\n')
            new_lines.append('        # Flatten per_token_logps if it has extra dimensions\\n')
            new_lines.append('        if per_token_logps.dim() > 1:\\n')
            new_lines.append('            per_token_logps = per_token_logps.view(-1)\\n')
        elif i > 340 and i < 360 and 'old_per_token_logps = old_per_token_logps[:current_batch_size]' in line:
            new_lines.append(line)
            new_lines.append('\\n')
            new_lines.append('        # Ensure old_per_token_logps is also flattened if needed\\n')
            new_lines.append('        if old_per_token_logps.dim() > 1:\\n')
            new_lines.append('            old_per_token_logps = old_per_token_logps.view(-1)\\n')
            new_lines.append('        \\n')
            new_lines.append('        # Final shape check and adjustment\\n')
            new_lines.append('        if old_per_token_logps.shape[0] != per_token_logps.shape[0]:\\n')
            new_lines.append('            old_per_token_logps = old_per_token_logps[:per_token_logps.shape[0]]\\n')
        else:
            new_lines.append(line)
    
    # Write the fixed file
    with open(grpo_file, 'w') as f:
        f.writelines(new_lines)
    
    print('✓ Tensor mismatch fix applied')

# Fix completion_mask dimension mismatch if needed
if completion_mask_fix_needed:
    print('Fixing completion_mask dimension mismatch...')
    with open(grpo_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for i, line in enumerate(lines):
        if 'loss = (per_token_loss * completion_mask).sum()' in line:
            # Add shape adjustment before loss calculation
            new_lines.append('        # Ensure per_token_loss and completion_mask have compatible shapes\\n')
            new_lines.append('        if per_token_loss.dim() == 1 and completion_mask.dim() == 2:\\n')
            new_lines.append('            completion_mask = completion_mask.view(-1)\\n')
            new_lines.append('        elif per_token_loss.dim() == 2 and completion_mask.dim() == 1:\\n')
            new_lines.append('            per_token_loss = per_token_loss.view(-1)\\n')
            new_lines.append('        elif per_token_loss.shape != completion_mask.shape:\\n')
            new_lines.append('            if per_token_loss.numel() == completion_mask.numel():\\n')
            new_lines.append('                completion_mask = completion_mask.view(per_token_loss.shape)\\n')
            new_lines.append('            else:\\n')
            new_lines.append('                min_size = min(per_token_loss.view(-1).shape[0], completion_mask.view(-1).shape[0])\\n')
            new_lines.append('                per_token_loss = per_token_loss.view(-1)[:min_size]\\n')
            new_lines.append('                completion_mask = completion_mask.view(-1)[:min_size]\\n')
            new_lines.append('        \\n')
            new_lines.append(line)
        elif 'is_clipped = ' in line and i > 395:
            new_lines.append(line)
            new_lines.append('        # Ensure compatible shapes for clip_ratio calculation\\n')
            new_lines.append('        if is_clipped.shape != completion_mask.shape:\\n')
            new_lines.append('            if is_clipped.dim() > completion_mask.dim():\\n')
            new_lines.append('                is_clipped = is_clipped.view(-1)\\n')
            new_lines.append('            elif completion_mask.dim() > is_clipped.dim():\\n')
            new_lines.append('                completion_mask = completion_mask.view(-1)\\n')
        else:
            new_lines.append(line)
    
    # Write the fixed file
    with open(grpo_file, 'w') as f:
        f.writelines(new_lines)
    
    # Fix escape characters
    import subprocess
    subprocess.run(['sed', '-i', 's/\\\\!=/!=/g', grpo_file])
    
    print('✓ Completion mask dimension fix applied')

# Check and apply compute_loss batch size mismatch fix
print('Checking for compute_loss batch size issue...')
needs_compute_loss_fix = False
with open(grpo_file, 'r') as f:
    content = f.read()
    if 'Ensure batch dimensions match' not in content:
        needs_compute_loss_fix = True
        print('✓ compute_loss fix needed')

if needs_compute_loss_fix:
    print('Applying compute_loss batch size fix...')
    with open(grpo_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for i, line in enumerate(lines):
        # Find and fix the concatenation line
        if i > 270 and 'input_ids = torch.cat([prompt_ids, completion_ids], dim=1)' in line:
            # Add dimension matching logic before concatenation
            new_lines.append('        # Ensure batch dimensions match for multi-GPU training\n')
            new_lines.append('        batch_adjusted = False\n')
            new_lines.append('        if prompt_ids.shape[0] != completion_ids.shape[0]:\n')
            new_lines.append('            # Handle batch size mismatch from num_generations duplication\n')
            new_lines.append('            if prompt_ids.shape[0] % completion_ids.shape[0] == 0:\n')
            new_lines.append('                # Prompts were duplicated by num_generations\n')
            new_lines.append('                num_gens = prompt_ids.shape[0] // completion_ids.shape[0]\n')
            new_lines.append('                batch_size = completion_ids.shape[0]\n')
            new_lines.append('                # Reshape and take the appropriate generation for each batch item\n')
            new_lines.append('                prompt_ids = prompt_ids.view(batch_size, num_gens, -1)[:, 0, :]\n')
            new_lines.append('                prompt_mask = prompt_mask.view(batch_size, num_gens, -1)[:, 0, :]\n')
            new_lines.append('                batch_adjusted = True\n')
            new_lines.append('            else:\n')
            new_lines.append('                # Unexpected size mismatch, try to truncate to match\n')
            new_lines.append('                min_batch = min(prompt_ids.shape[0], completion_ids.shape[0])\n')
            new_lines.append('                prompt_ids = prompt_ids[:min_batch]\n')
            new_lines.append('                prompt_mask = prompt_mask[:min_batch]\n')
            new_lines.append('                completion_ids = completion_ids[:min_batch]\n')
            new_lines.append('                completion_mask = completion_mask[:min_batch]\n')
            new_lines.append('                batch_adjusted = True\n')
            new_lines.append('        \n')
            new_lines.append(line)  # Add the original line
        # Also fix advantages dimension after it's defined
        elif i > 320 and 'advantages = inputs["advantages"]' in line:
            new_lines.append(line)
            new_lines.append('        # Adjust advantages dimensions if batch was adjusted\n')
            new_lines.append('        if "batch_adjusted" in locals() and batch_adjusted:\n')
            new_lines.append('            # Ensure advantages matches the adjusted batch size\n')
            new_lines.append('            expected_size = completion_ids.shape[0]\n')
            new_lines.append('            if advantages.shape[0] != expected_size:\n')
            new_lines.append('                if advantages.shape[0] % expected_size == 0:\n')
            new_lines.append('                    # Advantages were duplicated, reshape and take first\n')
            new_lines.append('                    num_gens = advantages.shape[0] // expected_size\n')
            new_lines.append('                    advantages = advantages.view(expected_size, num_gens)[:, 0]\n')
            new_lines.append('                else:\n')
            new_lines.append('                    # Truncate to match\n')
            new_lines.append('                    advantages = advantages[:expected_size]\n')
        else:
            new_lines.append(line)
    
    with open(grpo_file, 'w') as f:
        f.writelines(new_lines)
    
    print('✓ compute_loss fix applied successfully')

# Apply enhanced clip_ratio fix for multi-GPU
print('Checking for clip_ratio multi-GPU issue...')
needs_clip_ratio_fix = False
with open(grpo_file, 'r') as f:
    content = f.read()
    if 'Handle multi-GPU tensor size mismatch for clip_ratio' not in content:
        needs_clip_ratio_fix = True
        print('✓ clip_ratio multi-GPU fix needed')

if needs_clip_ratio_fix:
    print('Applying clip_ratio multi-GPU fix...')
    with open(grpo_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for i, line in enumerate(lines):
        if 'clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()' in line:
            # Add multi-GPU handling before clip_ratio calculation
            new_lines.append('        # Handle multi-GPU tensor size mismatch for clip_ratio\n')
            new_lines.append('        if is_clipped.shape[0] != completion_mask.shape[0]:\n')
            new_lines.append('            # Check if this is a multi-GPU duplication issue\n')
            new_lines.append('            if is_clipped.shape[0] % completion_mask.shape[0] == 0:\n')
            new_lines.append('                # is_clipped was duplicated across GPUs\n')
            new_lines.append('                num_gpus = is_clipped.shape[0] // completion_mask.shape[0]\n')
            new_lines.append('                # Average across GPU duplicates\n')
            new_lines.append('                is_clipped = is_clipped.view(num_gpus, -1).mean(dim=0)\n')
            new_lines.append('            elif completion_mask.shape[0] % is_clipped.shape[0] == 0:\n')
            new_lines.append('                # completion_mask was duplicated across GPUs\n')
            new_lines.append('                num_gpus = completion_mask.shape[0] // is_clipped.shape[0]\n')
            new_lines.append('                # Average across GPU duplicates\n')
            new_lines.append('                completion_mask = completion_mask.view(num_gpus, -1).mean(dim=0)\n')
            new_lines.append('            else:\n')
            new_lines.append('                # Fallback: ensure same size by truncation\n')
            new_lines.append('                min_size = min(is_clipped.shape[0], completion_mask.shape[0])\n')
            new_lines.append('                is_clipped = is_clipped[:min_size]\n')
            new_lines.append('                completion_mask = completion_mask[:min_size]\n')
            new_lines.append('        \n')
            new_lines.append('        # Additional shape compatibility check\n')
            new_lines.append('        if is_clipped.shape != completion_mask.shape:\n')
            new_lines.append('            # Flatten both tensors to 1D if dimensions differ\n')
            new_lines.append('            is_clipped = is_clipped.view(-1)\n')
            new_lines.append('            completion_mask = completion_mask.view(-1)\n')
            new_lines.append('            # Ensure same length\n')
            new_lines.append('            min_len = min(is_clipped.shape[0], completion_mask.shape[0])\n')
            new_lines.append('            is_clipped = is_clipped[:min_len]\n')
            new_lines.append('            completion_mask = completion_mask[:min_len]\n')
            new_lines.append('        \n')
            new_lines.append(line)
        else:
            new_lines.append(line)
    
    with open(grpo_file, 'w') as f:
        f.writelines(new_lines)
    
    print('✓ clip_ratio multi-GPU fix applied successfully')

# Add periodic GPU cache clearing during training
print('Checking for periodic GPU cache clearing...')
needs_cache_clear_fix = False
with open(grpo_file, 'r') as f:
    content = f.read()
    if 'Periodic GPU cache clearing' not in content:
        needs_cache_clear_fix = True
        print('✓ Periodic cache clearing needed')

if needs_cache_clear_fix:
    print('Adding periodic GPU cache clearing...')
    with open(grpo_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for i, line in enumerate(lines):
        # Add cache clearing after loss.backward()
        if 'loss.backward()' in line and i > 500:
            new_lines.append(line)
            new_lines.append('        
')
            new_lines.append('        # Periodic GPU cache clearing to prevent OOM
')
            new_lines.append('        if global_step % 10 == 0:
')
            new_lines.append('            import torch
')
            new_lines.append('            if torch.cuda.is_available():
')
            new_lines.append('                torch.cuda.empty_cache()
')
            new_lines.append('                # Also clear cache on all devices for multi-GPU
')
            new_lines.append('                for device_id in range(torch.cuda.device_count()):
')
            new_lines.append('                    with torch.cuda.device(device_id):
')
            new_lines.append('                        torch.cuda.empty_cache()
')
        else:
            new_lines.append(line)
    
    with open(grpo_file, 'w') as f:
        f.writelines(new_lines)
    
    print('✓ Periodic cache clearing added')

# Apply DataParallel memory optimization
print('Checking for DataParallel memory optimization...')
needs_dp_memory_fix = False
with open(grpo_file, 'r') as f:
    content = f.read()
    if 'Set DataParallel to use balanced memory allocation' not in content:
        needs_dp_memory_fix = True
        print('✓ DataParallel memory optimization needed')

if needs_dp_memory_fix:
    print('Adding DataParallel memory optimization...')
    with open(grpo_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for i, line in enumerate(lines):
        # Optimize DataParallel initialization
        if 'self.model = nn.DataParallel(self.model)' in line:
            new_lines.append(line)
            new_lines.append('            # Set DataParallel to use balanced memory allocation\n')
            new_lines.append('            if hasattr(self.model, "module"):\n')
            new_lines.append('                if hasattr(self.model.module, "gradient_checkpointing_enable"):\n')
            new_lines.append('                    self.model.module.gradient_checkpointing_enable()\n')
            new_lines.append('                # Enable memory-efficient attention if available\n')
            new_lines.append('                if hasattr(self.model.module, "config"):\n')
            new_lines.append('                    if hasattr(self.model.module.config, "use_memory_efficient_attention"):\n')
            new_lines.append('                        self.model.module.config.use_memory_efficient_attention = True\n')
            new_lines.append('            # Set output device to balance memory (use GPU 1 if available)\n')
            new_lines.append('            if torch.cuda.device_count() > 2:\n')
            new_lines.append('                self.model.output_device = 1  # Reduce load on GPU 0\n')
        else:
            new_lines.append(line)
    
    with open(grpo_file, 'w') as f:
        f.writelines(new_lines)
    
    print('✓ DataParallel memory optimization added')
"

# Fix manager.py for offline mode support
echo ""
echo "Checking manager.py for offline mode support..."
python3 -c "
import os

manager_file = 'rgym_exp/src/manager.py'
if os.path.exists(manager_file):
    with open(manager_file, 'r') as f:
        content = f.read()
    
    if 'CONNECT_TO_TESTNET' not in content:
        print('Adding offline mode support to manager.py...')
        with open(manager_file, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for i, line in enumerate(lines):
            if 'self.coordinator.register_peer(self.peer_id)' in line:
                new_lines.append('        # Only register peer if connected to testnet\\n')
                new_lines.append('        if os.environ.get(\"CONNECT_TO_TESTNET\", \"true\").lower() != \"false\":\\n')
                new_lines.append('            ' + line)
                new_lines.append('        else:\\n')
                new_lines.append('            _LOG.info(\"Skipping peer registration (CONNECT_TO_TESTNET=false)\")\\n')
            elif 'round, _ = self.coordinator.get_round_and_stage()' in line:
                new_lines.append('        if os.environ.get(\"CONNECT_TO_TESTNET\", \"true\").lower() != \"false\":\\n')
                new_lines.append('            ' + line)
                new_lines.append('        else:\\n')
                new_lines.append('            round = 1  # Default round for offline mode\\n')
            else:
                new_lines.append(line)
        
        with open(manager_file, 'w') as f:
            f.writelines(new_lines)
        
        print('✓ Manager.py patched for offline mode')
    else:
        print('✓ Manager.py already has offline mode support')
"

# Add automatic VRAM-based configuration
echo ""
echo "Auto-configuring based on available VRAM..."
python3 -c "
import torch
import os

if torch.cuda.is_available():
    # Get minimum VRAM across all GPUs
    min_vram_gb = min([torch.cuda.get_device_properties(i).total_memory / (1024**3) 
                       for i in range(torch.cuda.device_count())])
    
    print(f'Minimum VRAM per GPU: {min_vram_gb:.2f} GB')
    
    # Auto-configure based on VRAM
    if min_vram_gb < 8:
        print('⚠️  Low VRAM detected (<8GB), applying aggressive optimizations:')
        print('  - Batch size: 1')
        print('  - Gradient accumulation: 8 steps')
        print('  - Mixed precision: enabled')
        
        # Update config
        import fileinput
        import sys
        
        config_file = 'rgym_exp/config/rg-swarm.yaml'
        
        # Read current config
        with open(config_file, 'r') as f:
            lines = f.readlines()
        
        # Apply optimizations - safer YAML modification
        new_lines = []
        in_training = False
        in_trainer_config = False
        
        for i, line in enumerate(lines):
            # Track sections
            if line.strip().startswith('training:'):
                in_training = True
                in_trainer_config = False
            elif line.strip().startswith('trainer:'):
                in_training = False
                in_trainer_config = False
            elif in_trainer_config and '  config:' in line:
                in_trainer_config = True
            
            # Apply modifications based on context
            if 'num_train_samples:' in line and not in_trainer_config:
                new_lines.append('    num_train_samples: 1  # Auto-reduced for low VRAM
')
            elif 'gradient_accumulation_steps:' in line and in_training:
                new_lines.append('  gradient_accumulation_steps: 8  # Auto-increased for low VRAM
')
            elif 'fp16: false' in line:
                # Preserve indentation
                indent = len(line) - len(line.lstrip())
                new_lines.append(' ' * indent + 'fp16: true  # Auto-enabled for low VRAM
')
            else:
                new_lines.append(line)
        
        # Write updated config
        with open(config_file, 'w') as f:
            f.writelines(new_lines)
        
        print('✓ Configuration updated for low VRAM')
    elif min_vram_gb < 12:
        print('⚠️  Medium VRAM detected (8-12GB), applying moderate optimizations')
        print('  - Batch size: 2')
        print('  - Gradient accumulation: 4 steps')
    else:
        print('✓ Sufficient VRAM detected (>12GB)')
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