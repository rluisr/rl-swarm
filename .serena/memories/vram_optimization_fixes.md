# VRAM Optimization Fixes for Multi-GPU Training

## Problem
CUDA out of memory errors with unbalanced GPU memory usage:
- GPU 0: 7.63GB/7.66GB (nearly full)
- GPU 1: 7816MiB used
- GPU 2: 5196MiB used

## Implemented Solutions

### 1. Environment Variables
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True`
- `PYTORCH_NO_CUDA_MEMORY_CACHING=0`

### 2. Automatic VRAM-Based Configuration
Detects available VRAM and auto-configures:
- <8GB: batch_size=1, gradient_accumulation=8, fp16=true
- 8-12GB: batch_size=2, gradient_accumulation=4
- >12GB: default settings

### 3. Periodic GPU Cache Clearing
- Clears cache every 10 steps during training
- Clears cache on all GPUs in multi-GPU setup

### 4. DataParallel Memory Optimization
- Enables gradient checkpointing on module
- Uses memory-efficient attention when available
- Sets output_device=1 for >2 GPUs to reduce GPU 0 load

### 5. Multi-GPU Tensor Size Fixes
- Handles tensor size mismatches for clip_ratio calculation
- Manages GPU duplication with averaging/truncation
- Ensures shape compatibility for all operations

## Files Modified
- `start_with_gpu_optimization.sh`: All optimizations
- `patches/fix_dataparallel_memory.patch`: Memory balance patch

## Usage
Run `./start_with_gpu_optimization.sh` to apply all optimizations automatically.