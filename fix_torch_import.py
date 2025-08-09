#!/usr/bin/env python3
"""
Fix torch import issue in grpo_trainer.py
"""
import os
import sys

def fix_torch_import():
    # Multiple possible paths for the grpo_trainer.py file
    possible_paths = [
        '/root/test/.venv/lib/python3.10/site-packages/genrl/trainer/grpo_trainer.py',
        '.venv/lib/python3.10/site-packages/genrl/trainer/grpo_trainer.py',
        'venv/lib/python3.10/site-packages/genrl/trainer/grpo_trainer.py',
    ]
    
    grpo_file = None
    for path in possible_paths:
        if os.path.exists(path):
            grpo_file = path
            print(f"Found grpo_trainer.py at: {path}")
            break
    
    if not grpo_file:
        print("ERROR: grpo_trainer.py not found in expected locations")
        print("Searched paths:")
        for path in possible_paths:
            print(f"  - {path}")
        return False
    
    # Read the file
    with open(grpo_file, 'r') as f:
        lines = f.readlines()
    
    # Check if torch is already imported at the top
    has_torch_import = False
    for i in range(min(50, len(lines))):  # Check first 50 lines
        if 'import torch' in lines[i] and not lines[i].strip().startswith('#'):
            has_torch_import = True
            print(f"✓ Found existing torch import at line {i+1}")
            break
    
    # Add torch import if not present
    modified = False
    new_lines = []
    
    if not has_torch_import:
        print("Adding torch import at the beginning of file...")
        # Find where to insert import (after other imports)
        import_line = 0
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('import') and not line.strip().startswith('from'):
                import_line = i
                break
            if 'import' in line or 'from' in line:
                import_line = i + 1
        
        # Insert torch import
        for i, line in enumerate(lines):
            if i == import_line and not has_torch_import:
                new_lines.append('import torch\n')
                modified = True
                print(f"✓ Added 'import torch' at line {i+1}")
            new_lines.append(line)
    else:
        new_lines = lines[:]
    
    # Also check for the specific error location and add local import if needed
    for i, line in enumerate(new_lines):
        if 'rewards = torch.tensor(rewards)' in line and i > 500:
            # Check if there's already a local import
            if i > 0 and 'import torch' not in new_lines[i-1]:
                # Insert local import before this line
                indent = '        '  # Assuming this is inside a method
                if line.startswith('    '):
                    # Count actual indentation
                    indent = ''
                    for char in line:
                        if char in ' \t':
                            indent += char
                        else:
                            break
                
                # Insert the import
                new_lines.insert(i, f'{indent}import torch  # Local import to ensure torch is available\n')
                modified = True
                print(f"✓ Added local torch import at line {i+1}")
                break
    
    if modified:
        # Backup original file
        backup_file = grpo_file + '.backup_torch'
        if not os.path.exists(backup_file):
            import shutil
            shutil.copy(grpo_file, backup_file)
            print(f"✓ Backup created: {backup_file}")
        
        # Write the modified file
        with open(grpo_file, 'w') as f:
            f.writelines(new_lines)
        
        print("✓ File successfully patched")
        return True
    else:
        print("✓ No modifications needed - torch import already present")
        return True

if __name__ == "__main__":
    success = fix_torch_import()
    sys.exit(0 if success else 1)