# GRPO Trainer Indentation Fix

## Problem
IndentationError at line 521 in grpo_trainer.py:
```
IndentationError: unexpected indent
```

## Root Cause
The `import torch` statement at line 521 had incorrect indentation (12 spaces instead of 8).

## Fix Applied
Changed line 521 from:
```python
            import torch  # Ensure torch is available in this scope  # 12 spaces
```
To:
```python
        import torch  # Ensure torch is available in this scope  # 8 spaces
```

## Location
`/root/test/.venv/lib/python3.10/site-packages/genrl/trainer/grpo_trainer.py:521`

## Result
The indentation error has been fixed. The import statement now has the correct indentation level matching the surrounding code block.