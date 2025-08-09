# CUDA_VISIBLE_DEVICES Multi-GPU Fix

## Problem
When running with `CUDA_VISIBLE_DEVICES=1,2`, the training was still trying to use GPU 0 and causing OOM errors.

## Root Cause
1. DataParallel was not respecting CUDA_VISIBLE_DEVICES environment variable
2. The model was not explicitly placed on the correct visible devices
3. Batch size was too large for available VRAM

## Solutions Applied

### 1. Fixed trainer.py DataParallel initialization
Modified `rgym_exp/src/trainer.py` to:
- Log CUDA_VISIBLE_DEVICES value
- Explicitly specify device_ids for DataParallel
- Move model to correct device before wrapping
- Handle single GPU case properly

### 2. Reduced batch size
Changed `num_train_samples` from 2 to 1 in `rgym_exp/config/rg-swarm.yaml`

### 3. Created helper scripts
- `run_with_gpu_1_2.sh`: Wrapper script that sets environment variables correctly
- `fix_gpu_oom.py`: Documentation of fixes

### 4. Environment variables
Set the following for memory optimization:
```bash
export CUDA_VISIBLE_DEVICES=1,2
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
```

## Usage
Run training with:
```bash
./run_with_gpu_1_2.sh
```

Or manually:
```bash
CUDA_VISIBLE_DEVICES=1,2 ./start_with_gpu_optimization.sh
```

## Important Notes
- When CUDA_VISIBLE_DEVICES=1,2 is set, PyTorch sees these as devices 0 and 1
- DataParallel must use device_ids=[0,1] in this case, not [1,2]
- The model should be moved to cuda:0 (which maps to physical GPU 1)