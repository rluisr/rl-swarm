#!/usr/bin/env python3
"""
GPU monitoring and memory management utilities for RL Swarm.
Helps prevent and handle CUDA OutOfMemory errors.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import logging
import os
import sys

logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitor and manage GPU resources for multi-GPU training."""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        
    def get_gpu_summary(self) -> str:
        """Get a formatted summary of all GPU status."""
        if not self.cuda_available:
            return "CUDA is not available"
        
        summary = f"Found {self.device_count} GPU(s):\n"
        for i in range(self.device_count):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            memory_total = props.total_memory / (1024**3)
            
            summary += f"  GPU {i}: {props.name}\n"
            summary += f"    Total Memory: {memory_total:.2f} GB\n"
            summary += f"    Allocated: {memory_allocated:.2f} GB\n"
            summary += f"    Reserved: {memory_reserved:.2f} GB\n"
            summary += f"    Free: {memory_total - memory_reserved:.2f} GB\n"
            
        return summary
    
    def check_memory_availability(self, required_gb: float = 2.0) -> bool:
        """Check if there's enough GPU memory available."""
        if not self.cuda_available:
            return False
        
        for i in range(self.device_count):
            props = torch.cuda.get_device_properties(i)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            memory_total = props.total_memory / (1024**3)
            free_memory = memory_total - memory_reserved
            
            if free_memory < required_gb:
                logger.warning(f"GPU {i} has only {free_memory:.2f}GB free, less than required {required_gb}GB")
                return False
        
        return True
    
    def optimize_memory_allocation(self) -> Dict[str, Any]:
        """Optimize memory allocation and return recommendations."""
        if not self.cuda_available:
            return {"success": False, "message": "CUDA not available"}
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        recommendations = {
            "success": True,
            "device_count": self.device_count,
            "recommendations": []
        }
        
        total_free_memory = 0
        for i in range(self.device_count):
            props = torch.cuda.get_device_properties(i)
            memory_total = props.total_memory / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            free_memory = memory_total - memory_reserved
            total_free_memory += free_memory
        
        # Calculate recommended batch size based on available memory
        # Assuming average model size of 1.5GB
        model_size_gb = 1.5
        overhead_gb = 2.0  # System overhead
        usable_memory = max(total_free_memory - overhead_gb, 1.0)
        
        # With DataParallel, memory is divided across GPUs
        recommended_batch_size = int(usable_memory / (model_size_gb * 2.5))
        recommended_batch_size = max(recommended_batch_size, 1)
        
        recommendations["recommendations"].append(
            f"Recommended total batch size: {recommended_batch_size}"
        )
        
        if self.device_count > 1:
            per_gpu_batch = max(recommended_batch_size // self.device_count, 1)
            recommendations["recommendations"].append(
                f"Per-GPU batch size: {per_gpu_batch}"
            )
            recommendations["recommendations"].append(
                "Using DataParallel for multi-GPU training"
            )
        
        # Memory optimization tips
        if total_free_memory < 8.0:
            recommendations["recommendations"].append(
                "Consider enabling gradient checkpointing (already enabled in config)"
            )
            recommendations["recommendations"].append(
                "Consider using fp16 training by setting fp16: true in config"
            )
        
        return recommendations


def wrap_model_for_multi_gpu(model: nn.Module, device_ids: Optional[List[int]] = None) -> nn.Module:
    """
    Wrap a model for multi-GPU training.
    
    Args:
        model: The model to wrap
        device_ids: List of GPU device IDs to use (None for all available)
    
    Returns:
        The wrapped model (DataParallel if multiple GPUs, original otherwise)
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, returning original model")
        return model
    
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        logger.info("Single GPU detected, using original model")
        return model.cuda()
    
    if isinstance(model, nn.DataParallel):
        logger.info("Model already wrapped with DataParallel")
        return model
    
    if device_ids is None:
        device_ids = list(range(device_count))
    
    logger.info(f"Wrapping model with DataParallel for GPUs: {device_ids}")
    model = nn.DataParallel(model, device_ids=device_ids)
    
    return model


def handle_cuda_oom(func):
    """
    Decorator to handle CUDA OutOfMemory errors gracefully.
    
    Usage:
        @handle_cuda_oom
        def train_step():
            # training code
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OutOfMemory error: {e}")
            
            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Get current GPU status
            monitor = GPUMonitor()
            logger.info("Current GPU status after OOM:")
            logger.info(monitor.get_gpu_summary())
            
            # Get recommendations
            recommendations = monitor.optimize_memory_allocation()
            for rec in recommendations.get("recommendations", []):
                logger.info(f"Recommendation: {rec}")
            
            raise RuntimeError(
                "CUDA OutOfMemory error occurred. Please reduce batch size or enable "
                "more memory optimizations. Check logs for recommendations."
            ) from e
    
    return wrapper


def main():
    """Main function for testing GPU utilities."""
    monitor = GPUMonitor()
    
    print("=== GPU Status ===")
    print(monitor.get_gpu_summary())
    
    print("\n=== Memory Optimization Recommendations ===")
    recommendations = monitor.optimize_memory_allocation()
    for rec in recommendations.get("recommendations", []):
        print(f"• {rec}")
    
    if monitor.cuda_available:
        print(f"\nMemory sufficient for 2GB requirement: {monitor.check_memory_availability(2.0)}")


if __name__ == "__main__":
    main()