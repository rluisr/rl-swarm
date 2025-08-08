# GPU Optimization for CUDA OutOfMemory Resolution

## Overview
This document describes the GPU optimizations implemented to resolve CUDA OutOfMemory errors in the RL Swarm project.

## Implemented Solutions

### 1. Multi-GPU Support (DataParallel)
- **File**: `rgym_exp/src/trainer.py`
- Automatically detects multiple GPUs and wraps the model with `nn.DataParallel`
- Distributes batch processing across all available GPUs
- Handles model attribute access for both wrapped and unwrapped models

### 2. Memory Optimization Settings
- **File**: `rgym_exp/config/rg-swarm.yaml`
- Added configuration options:
  - `gradient_checkpointing: true` - Reduces memory usage by trading computation
  - `gradient_accumulation_steps: 4` - Accumulates gradients over multiple steps
  - `clear_cache_every_n_rounds: 10` - Periodically clears GPU cache
  - `auto_batch_size: true` - Automatically adjusts batch size based on available memory

### 3. GPU Utility Functions
- **File**: `rgym_exp/src/utils/omega_gpu_resolver.py`
- Enhanced GPU detection to support multiple GPUs
- Added functions:
  - `get_gpu_info()` - Detailed GPU information including memory usage
  - `clear_gpu_cache()` - Clears GPU memory cache
  - `get_optimal_batch_size()` - Calculates optimal batch size based on available memory

### 4. GPU Monitoring Tool
- **File**: `rgym_exp/src/utils/gpu_monitor.py`
- Standalone GPU monitoring and management utility
- Features:
  - Real-time GPU memory status
  - Memory optimization recommendations
  - CUDA OOM error handling decorator
  - Multi-GPU wrapper function

### 5. Manager Updates
- **File**: `rgym_exp/src/manager.py`
- Added periodic GPU cache clearing
- Handles DataParallel wrapped models correctly
- Logs GPU status during training

### 6. Startup Script
- **File**: `start_with_gpu_optimization.sh`
- Interactive startup script with GPU optimization
- Features:
  - GPU availability check
  - Memory status display
  - Configuration options (fp16, batch size)
  - Automatic cache clearing

## Usage

### Quick Start
```bash
# Use the optimized startup script
./start_with_gpu_optimization.sh
```

### Manual Configuration
Edit `rgym_exp/config/rg-swarm.yaml`:

```yaml
training:
  fp16: true  # Enable for ~50% memory savings
  gradient_checkpointing: true  # Already enabled by default
  gradient_accumulation_steps: 4  # Increase for lower memory usage
  clear_cache_every_n_rounds: 10  # Adjust frequency as needed
```

### Monitor GPU Status
```bash
# Run the GPU monitor tool
python3 rgym_exp/src/utils/gpu_monitor.py
```

## Troubleshooting

### If OutOfMemory Still Occurs

1. **Reduce batch size**:
   ```yaml
   num_train_samples: 1  # Start with 1, increase gradually
   ```

2. **Enable fp16 training**:
   ```yaml
   fp16: true  # Significant memory savings
   ```

3. **Increase gradient accumulation**:
   ```yaml
   gradient_accumulation_steps: 8  # or higher
   ```

4. **Use smaller model**:
   - The system automatically selects models based on available VRAM
   - Models with <40GB VRAM use smaller variants

### Environment Variables
Set these before running:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
```

## Performance Tips

1. **Multi-GPU Setup**: The system automatically uses DataParallel when multiple GPUs are detected
2. **Memory Monitoring**: Check GPU status regularly with the monitoring tool
3. **Cache Management**: GPU cache is cleared every 10 rounds by default
4. **Batch Size**: Start small and increase gradually while monitoring memory

## Technical Details

### DataParallel Implementation
- Automatically wraps model when `torch.cuda.device_count() > 1`
- Splits input batch across GPUs
- Gathers outputs on primary GPU (GPU 0)
- ~Linear speedup with number of GPUs (some overhead)

### Memory Calculation
- Model size: ~1.5GB (estimated)
- Overhead: 2GB (PyTorch, system)
- Per-batch memory: model_size * 2.5 (includes gradients, optimizer states)
- Formula: `batch_size = (total_gpu_memory - overhead) / (model_size * 2.5)`

## Future Improvements
- Consider DistributedDataParallel for better multi-GPU efficiency
- Implement dynamic batch sizing based on real-time memory usage
- Add memory profiling tools
- Support for model parallelism for very large models