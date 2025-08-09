#!/usr/bin/env python3
"""Fix YAML indentation issues in rg-swarm.yaml"""

import sys

def fix_yaml(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    in_trainer_config = False
    config_section = False
    
    for i, line in enumerate(lines):
        # Detect trainer section
        if 'trainer:' in line and i+1 < len(lines) and '_target_' in lines[i+1]:
            in_trainer_config = True
            new_lines.append(line)
        # Detect config section under trainer
        elif in_trainer_config and 'config:' in line and '_target_' in lines[i+1] if i+1 < len(lines) else False:
            config_section = True
            new_lines.append(line)
        # Fix misplaced fp16 and other config options
        elif config_section and line.strip().startswith('fp16:'):
            new_lines.append('      fp16: true\n')
        elif config_section and line.strip().startswith('log_with:'):
            new_lines.append('      log_with: wandb\n')
        elif config_section and line.strip().startswith('log_dir:'):
            new_lines.append('      log_dir: ${log_dir}\n')
        elif config_section and line.strip().startswith('epsilon:'):
            new_lines.append('      epsilon: 0.2\n')
        elif config_section and line.strip().startswith('epsilon_high:'):
            new_lines.append('      epsilon_high: 0.28\n')
        elif config_section and line.strip().startswith('num_generations:'):
            new_lines.append('      num_generations: ${training.num_generations}\n')
        elif config_section and line.strip().startswith('judge_base_url:'):
            new_lines.append('      judge_base_url: ${eval.judge_base_url}\n')
        elif config_section and line.strip().startswith('gradient_checkpointing:'):
            new_lines.append('      gradient_checkpointing: ${training.gradient_checkpointing}\n')
        elif config_section and line.strip().startswith('gradient_accumulation_steps:'):
            new_lines.append('      gradient_accumulation_steps: ${training.gradient_accumulation_steps}\n')
        elif config_section and line.strip().startswith('num_train_samples:'):
            new_lines.append('      num_train_samples: 1  # Auto-reduced for low VRAM\n')
        # End of trainer section
        elif 'data_manager:' in line and '_target_' in lines[i+1] if i+1 < len(lines) else False:
            in_trainer_config = False
            config_section = False
            new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Write fixed file
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✓ Fixed YAML file: {file_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fix_yaml(sys.argv[1])
    else:
        print("Usage: python fix_yaml.py <path_to_yaml>")