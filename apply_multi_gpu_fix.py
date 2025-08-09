#!/usr/bin/env python3
"""
Apply multi-GPU fix for GRPO trainer compute_loss method.
This fixes the tensor size mismatch when using DataParallel.
"""

import sys

def apply_fix():
    file_path = "/root/test/.venv/lib/python3.10/site-packages/genrl/trainer/grpo_trainer.py"
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the line with "advantages = inputs["advantages"]"
    modified = False
    for i, line in enumerate(lines):
        if 'advantages = inputs["advantages"]' in line and not modified:
            # Insert the fix after this line
            indent = len(line) - len(line.lstrip())
            fix_lines = [
                line,  # Keep the original line
                ' ' * indent + '\n',
                ' ' * indent + '# Handle DataParallel: ensure advantages match the current batch size\n',
                ' ' * indent + '# This ensures compatibility when the model is wrapped in nn.DataParallel\n',
                ' ' * indent + 'current_batch_size = per_token_logps.shape[0]\n',
                ' ' * indent + 'if advantages.shape[0] != current_batch_size:\n',
                ' ' * (indent + 4) + '# Split advantages to match the current GPU\'s batch\n',
                ' ' * (indent + 4) + 'advantages = advantages[:current_batch_size]\n',
                ' ' * indent + '\n',
            ]
            # Replace the original line with our fixed version
            lines[i:i+1] = fix_lines
            modified = True
            break
    
    if not modified:
        print("Error: Could not find the line to patch")
        return False
    
    # Find and fix old_per_token_logps handling
    for i, line in enumerate(lines):
        if 'else per_token_logps.detach()' in line:
            # Add fix after the closing parenthesis
            if i + 1 < len(lines) and ')' in lines[i+1]:
                insert_idx = i + 2
                indent = len(lines[i+2]) - len(lines[i+2].lstrip())
                fix_lines2 = [
                    '\n',
                    ' ' * indent + '# Also handle old_per_token_logps for DataParallel\n',
                    ' ' * indent + 'if self.args.num_iterations > 1 and old_per_token_logps.shape[0] != current_batch_size:\n',
                    ' ' * (indent + 4) + 'old_per_token_logps = old_per_token_logps[:current_batch_size]\n',
                    '\n',
                ]
                lines[insert_idx:insert_idx] = fix_lines2
                break
    
    # Write the modified file
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print("Fix applied successfully!")
    return True

if __name__ == "__main__":
    if apply_fix():
        sys.exit(0)
    else:
        sys.exit(1)