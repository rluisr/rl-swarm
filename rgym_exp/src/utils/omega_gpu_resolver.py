import random

import torch
from omegaconf import OmegaConf


def get_gpu_vram():
    """Returns the total VRAM of all available GPUs in GiB."""
    if not torch.cuda.is_available():
        return 0
    
    device_count = torch.cuda.device_count()
    total_vram = 0
    
    for i in range(device_count):
        device_vram = torch.cuda.get_device_properties(i).total_memory
        total_vram += device_vram
    
    return total_vram / (1024**3)  # Convert bytes to GiB  # Convert bytes to GiB

def get_gpu_info():
    """Returns detailed GPU information including memory usage."""
    if not torch.cuda.is_available():
        return {"available": False, "devices": []}
    
    gpu_info = {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "devices": []
    }
    
    for i in range(gpu_info["device_count"]):
        props = torch.cuda.get_device_properties(i)
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
        memory_total = props.total_memory / (1024**3)
        
        device_info = {
            "index": i,
            "name": props.name,
            "total_memory_gb": memory_total,
            "allocated_memory_gb": memory_allocated,
            "reserved_memory_gb": memory_reserved,
            "free_memory_gb": memory_total - memory_reserved,
            "compute_capability": f"{props.major}.{props.minor}"
        }
        gpu_info["devices"].append(device_info)
    
    return gpu_info


def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return True
    return False


def get_optimal_batch_size(model_size_gb=1.5, available_memory_gb=None):
    """
    Calculate optimal batch size based on available GPU memory.
    
    Args:
        model_size_gb: Estimated model size in GB
        available_memory_gb: Available GPU memory in GB (auto-detected if None)
    
    Returns:
        Suggested batch size
    """
    if available_memory_gb is None:
        gpu_info = get_gpu_info()
        if not gpu_info["available"] or not gpu_info["devices"]:
            return 1
        
        # Use the minimum free memory across all GPUs for safety
        available_memory_gb = min(d["free_memory_gb"] for d in gpu_info["devices"])
    
    # Reserve 2GB for system/PyTorch overhead
    usable_memory = max(available_memory_gb - 2, 1)
    
    # Rough estimation: batch_size = usable_memory / (model_size * 2.5)
    # The 2.5 factor accounts for gradients, optimizer states, and activations
    batch_size = int(usable_memory / (model_size_gb * 2.5))
    
    # Ensure minimum batch size of 1
    return max(batch_size, 1)


def gpu_model_choice_resolver(large_model_pool, small_model_pool):
    """Selects a model from the large or small pool based on VRAM."""
    vram = get_gpu_vram()
    if vram >= 40:
        model_pool = large_model_pool
    else:
        model_pool = small_model_pool
    return random.choice(model_pool)


OmegaConf.register_new_resolver("gpu_model_choice", gpu_model_choice_resolver)
