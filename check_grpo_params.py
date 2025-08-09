#!/usr/bin/env python3
"""Check valid GRPOConfig parameters"""

try:
    from trl.trainer import GRPOConfig
    import inspect
    
    # Get valid parameters
    params = inspect.signature(GRPOConfig.__init__).parameters
    valid_params = [k for k in params.keys() if k != 'self']
    
    print("Valid GRPOConfig parameters:")
    for param in valid_params[:20]:  # Show first 20
        print(f"  - {param}")
    
    # Check if our parameters are valid
    our_params = [
        'logging_dir', 'fp16', 'epsilon', 'epsilon_high', 
        'num_generations', 'gradient_checkpointing', 
        'gradient_accumulation_steps', 'num_train_samples'
    ]
    
    print("\nChecking our parameters:")
    for param in our_params:
        if param in valid_params:
            print(f"  ✓ {param} - valid")
        else:
            print(f"  ✗ {param} - INVALID")
            
except Exception as e:
    print(f"Error: {e}")
    print("Could not import GRPOConfig")