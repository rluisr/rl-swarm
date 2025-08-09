# Torch Import Fix for grpo_trainer.py

## Problem
UnboundLocalError: local variable 'torch' referenced before assignment at line 521 in grpo_trainer.py

## Root Cause
The torch module was not imported in the local scope where it was being used at line 521:
```python
rewards = torch.tensor(rewards)
```

## Solution Applied
Added a local import statement just before the problematic line:
```python
import torch  # Ensure torch is available in this scope
rewards = torch.tensor(rewards)
```

## Files Modified
- `/root/test/.venv/lib/python3.10/site-packages/genrl/trainer/grpo_trainer.py`
  - Added local `import torch` at line 521
  - Backup created: `grpo_trainer.py.backup_torch`

## Fix Script Location
- `/Users/s01082/ghq/github.com/gensyn-ai/rl-swarm/fix_torch_import.py`

## Application Method
Applied via SSH to remote server:
```bash
ssh hiveos.luis.local -l user "sudo python3 - << 'EOF'
# Python script content
EOF"
```

## Verification
The fix was successfully applied and verified. The torch import is now present at line 521, ensuring torch is available in the local scope where it's needed.