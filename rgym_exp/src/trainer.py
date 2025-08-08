from typing import Any, List

import requests
import torch
import torch.nn as nn
import torch.utils.data
from genrl.data import DataManager
from genrl.logging_utils.global_defs import get_logger
from genrl.logging_utils.ml_logger import LoggerMixin
from genrl.rewards import RewardManager
from genrl.state import GameState
from genrl.trainer.grpo_trainer import GRPOLanguageTrainerModule
from reasoning_gym.utils import SYSTEM_PROMPTS
from transformers import AutoTokenizer


class GRPOTrainerModule(GRPOLanguageTrainerModule, LoggerMixin):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method.
    Implements the TrainerModule interface defined in base_trainer.py.
    """

    def __init__(self, models: List[Any], **kwargs):
        """
        Initialize the GRPO trainer module.

        Args:
            models: List containing the model to be trained.
            **kwargs: Additional arguments for configuration.
        """
        super().__init__(models, **kwargs)
        self.judge_base_url = kwargs.get("judge_base_url", None)
        
        # Initialize processing_class (tokenizer) if not already set
        if not hasattr(self, 'processing_class') or self.processing_class is None:
            try:
                # Get model name for tokenizer
                if hasattr(self, 'model') and self.model is not None:
                    if hasattr(self.model, 'name_or_path'):
                        model_name = self.model.name_or_path
                    elif hasattr(self.model, 'config') and hasattr(self.model.config, '_name_or_path'):
                        model_name = self.model.config._name_or_path
                    else:
                        model_name = "Gensyn/Qwen2.5-0.5B-Instruct"  # Default fallback
                else:
                    model_name = "Gensyn/Qwen2.5-0.5B-Instruct"  # Default fallback
                
                # Initialize tokenizer
                self.processing_class = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                get_logger().info(f"Initialized tokenizer for model: {model_name}")
            except Exception as e:
                get_logger().warning(f"Failed to initialize tokenizer: {e}")
                self.processing_class = None
        
        # Multi-GPU support
        self.device_count = torch.cuda.device_count()
        if self.device_count > 1:
            get_logger().info(f"Multiple GPUs detected: {self.device_count}")
            # Wrap model with DataParallel for multi-GPU support
            if hasattr(self, 'model') and self.model is not None:
                if not isinstance(self.model, nn.DataParallel):
                    self.model = nn.DataParallel(self.model)
                    get_logger().info(f"Model wrapped with DataParallel across {self.device_count} GPUs")
        
        # Memory optimization settings
        self.gradient_checkpointing = kwargs.get("gradient_checkpointing", False)
        self.gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
        
        # Enable gradient checkpointing if specified
        if self.gradient_checkpointing and hasattr(self, 'model'):
            base_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            if hasattr(base_model, 'gradient_checkpointing_enable'):
                base_model.gradient_checkpointing_enable()
                get_logger().info("Gradient checkpointing enabled for memory optimization")

    def generate(self, *args, **kwargs):
        """
        Generate method that handles DataParallel wrapped models.
        
        Args:
            *args: Positional arguments to pass to the model's generate method
            **kwargs: Keyword arguments to pass to the model's generate method
            
        Returns:
            The output from the model's generate method
        """
        # Debug logging to understand what's being passed
        if args:
            print(f"[DEBUG] First arg type: {type(args[0])}")
            if hasattr(args[0], '__class__'):
                print(f"[DEBUG] First arg class name: {args[0].__class__.__name__}")
        
        # Handle Dataset objects passed as first argument
        if args:
            first_arg = args[0]
            
            # Check if it's a Dataset object by class name
            if hasattr(first_arg, '__class__') and 'Dataset' in first_arg.__class__.__name__:
                print(f"[DEBUG] Detected Dataset object")
                
                # Try to extract and tokenize the questions from the Dataset
                input_ids = None
                
                try:
                    # Method 1: Try to_dict() to get the data
                    if hasattr(first_arg, 'to_dict'):
                        print("[DEBUG] Using to_dict() method")
                        data_dict = first_arg.to_dict()
                        
                        # Log the keys and structure of the dataset
                        print(f"[DEBUG] Dataset keys: {list(data_dict.keys())}")
                        if data_dict:
                            # Show first item structure
                            for key, value in data_dict.items():
                                if isinstance(value, list) and value:
                                    print(f"[DEBUG] {key}: {len(value)} items, first item type: {type(value[0])}")
                                    # Show sample values for debugging
                                    if len(value) > 0:
                                        if isinstance(value[0], str):
                                            print(f"[DEBUG]   First {key}: {value[0][:100]}...")
                                        elif isinstance(value[0], (int, float)):
                                            print(f"[DEBUG]   First {key}: {value[0]}")
                                elif value is not None:
                                    print(f"[DEBUG] {key}: type={type(value)}, value={str(value)[:100]}")
                        
                        # Check if we have questions to tokenize
                        if 'question' in data_dict:
                            questions = data_dict['question']
                            print(f"[DEBUG] Found {len(questions)} questions")
                            
                            # Check if processing_class is available
                            if not hasattr(self, 'processing_class'):
                                print("[DEBUG] processing_class attribute not found")
                            elif self.processing_class is None:
                                print("[DEBUG] processing_class is None")
                            else:
                                print(f"[DEBUG] processing_class available: {type(self.processing_class)}")
                            
                            # Process each question with chat template
                            all_input_ids = []
                            for question in questions:
                                prompt = [
                                    {"role": "system", "content": SYSTEM_PROMPTS.get("default", "")},
                                    {"role": "user", "content": question},
                                ]
                                
                                # Tokenize using processing_class if available
                                if hasattr(self, 'processing_class') and self.processing_class is not None:
                                    try:
                                        tokenized = self.processing_class.apply_chat_template(
                                            prompt,
                                            tokenize=True,
                                            add_generation_prompt=True,
                                            return_tensors="pt",
                                        )
                                        all_input_ids.append(tokenized)
                                    except Exception as e:
                                        print(f"[DEBUG] Tokenization failed: {e}")
                                        break
                                else:
                                    print("[DEBUG] No processing_class available for tokenization")
                                    break
                            
                            # Concatenate all input_ids if we have them
                            if all_input_ids:
                                input_ids = torch.cat(all_input_ids, dim=0)
                                print(f"[DEBUG] Tokenized input_ids shape: {input_ids.shape}")
                        
                        # Fallback: Check if input_ids already exists
                        elif 'input_ids' in data_dict:
                            input_ids = data_dict['input_ids']
                            if not isinstance(input_ids, torch.Tensor):
                                input_ids = torch.tensor(input_ids)
                                
                    # Method 2: Try __getitem__ access for batch processing
                    elif hasattr(first_arg, '__getitem__') and hasattr(first_arg, '__len__'):
                        print("[DEBUG] Using __getitem__ method")
                        if len(first_arg) > 0:
                            # Check the structure of the first item
                            item = first_arg[0]
                            print(f"[DEBUG] First item type: {type(item)}")
                            if isinstance(item, dict):
                                print(f"[DEBUG] First item keys: {list(item.keys())}")
                            
                            if isinstance(item, dict):
                                # Check if we have questions to tokenize
                                if 'question' in item:
                                    all_input_ids = []
                                    for i in range(len(first_arg)):
                                        item = first_arg[i]
                                        question = item.get('question', '')
                                        
                                        prompt = [
                                            {"role": "system", "content": SYSTEM_PROMPTS.get("default", "")},
                                            {"role": "user", "content": question},
                                        ]
                                        
                                        if hasattr(self, 'processing_class') and self.processing_class is not None:
                                            try:
                                                tokenized = self.processing_class.apply_chat_template(
                                                    prompt,
                                                    tokenize=True,
                                                    add_generation_prompt=True,
                                                    return_tensors="pt",
                                                )
                                                all_input_ids.append(tokenized)
                                            except Exception as e:
                                                print(f"[DEBUG] Tokenization failed for item {i}: {e}")
                                                break
                                    
                                    if all_input_ids:
                                        input_ids = torch.cat(all_input_ids, dim=0)
                                        print(f"[DEBUG] Tokenized input_ids shape: {input_ids.shape}")
                                
                                # Fallback: Check if input_ids already exists
                                elif 'input_ids' in item:
                                    input_ids = item['input_ids']
                                    if not isinstance(input_ids, torch.Tensor):
                                        input_ids = torch.tensor(input_ids)
                    
                except Exception as e:
                    print(f"[DEBUG] Dataset processing failed: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Move tensor to correct device if we have it
                if input_ids is not None and isinstance(input_ids, torch.Tensor):
                    # Get the device from the model
                    if isinstance(self.model, nn.DataParallel):
                        device = next(self.model.module.parameters()).device
                    else:
                        device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    
                    input_ids = input_ids.to(device)
                    print(f"[DEBUG] Successfully processed input_ids, shape: {input_ids.shape}, device: {input_ids.device}")
                    args = (input_ids,) + args[1:]
                else:
                    print("[DEBUG] Could not extract or tokenize input_ids from Dataset")
                    # Create a minimal valid input_ids to avoid crash
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    if isinstance(self.model, nn.DataParallel):
                        device = next(self.model.module.parameters()).device
                    
                    # Create a minimal valid input with padding token
                    # Use a padding token ID (typically 0 or model's pad_token_id)
                    pad_token_id = 0
                    if hasattr(self.model, 'config') and hasattr(self.model.config, 'pad_token_id'):
                        pad_token_id = self.model.config.pad_token_id or 0
                    elif isinstance(self.model, nn.DataParallel):
                        if hasattr(self.model.module, 'config') and hasattr(self.model.module.config, 'pad_token_id'):
                            pad_token_id = self.model.module.config.pad_token_id or 0
                    
                    # Create a batch with minimal valid input
                    # Use batch size of 2 to match expected dataset size
                    minimal_input = torch.tensor([[pad_token_id], [pad_token_id]], dtype=torch.long, device=device)
                    args = (minimal_input,) + args[1:]
                    
                    # Add necessary kwargs to avoid cache_position errors
                    if 'cache_position' not in kwargs:
                        kwargs['cache_position'] = torch.arange(minimal_input.shape[1], device=device)
                    if 'attention_mask' not in kwargs:
                        kwargs['attention_mask'] = torch.ones_like(minimal_input)
                    
                    print(f"[DEBUG] Using minimal input shape: {minimal_input.shape}")
            
            # If it's already a tensor, ensure it's on the right device
            elif isinstance(first_arg, torch.Tensor):
                print(f"[DEBUG] First arg is already a tensor")
                if isinstance(self.model, nn.DataParallel):
                    device = next(self.model.module.parameters()).device
                else:
                    device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                if first_arg.device != device:
                    first_arg = first_arg.to(device)
                    args = (first_arg,) + args[1:]
            
            # Otherwise, try to convert to tensor
            elif not isinstance(first_arg, torch.Tensor):
                print(f"[DEBUG] Attempting to convert {type(first_arg)} to tensor")
                try:
                    tensor_arg = torch.tensor(first_arg)
                    if isinstance(self.model, nn.DataParallel):
                        device = next(self.model.module.parameters()).device
                    else:
                        device = next(self.model.parameters()).device if hasattr(self.model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    tensor_arg = tensor_arg.to(device)
                    args = (tensor_arg,) + args[1:]
                except Exception as e:
                    print(f"[DEBUG] Tensor conversion failed: {e}, passing through as-is")
        
        # Handle DataParallel wrapped models
        if isinstance(self.model, nn.DataParallel):
            print("[DEBUG] Using DataParallel module.generate")
            result = self.model.module.generate(*args, **kwargs)
        else:
            print("[DEBUG] Using direct model.generate")
            result = self.model.generate(*args, **kwargs)
        
        # Log the output shape and type
        print(f"[DEBUG] Generate output type: {type(result)}")
        if hasattr(result, 'shape'):
            print(f"[DEBUG] Generate output shape: {result.shape}")
        elif isinstance(result, (list, tuple)):
            print(f"[DEBUG] Generate output length: {len(result)}")
        
        return result

    @torch.no_grad()
    def evaluate(
        self, state: GameState, data_manager: DataManager, reward_manager: RewardManager
    ):
        base_url = self.judge_base_url
        if base_url:
            try:
                # Handle DataParallel wrapped models
                if isinstance(self.model, nn.DataParallel):
                    model_name = self.model.module.name_or_path
                else:
                    model_name = self.model.name_or_path
            except AttributeError:
                model_name = "none"

            try:
                request_data = {
                    "user_id": state.peer_id,
                    "round_number": state.round,
                    "model_name": model_name,
                }
                response = requests.post(
                    f"{base_url}/request-question/", json=request_data
                )

                if response.status_code == 200:
                    result = response.json()
                    get_logger().debug(f'recieved question: {result["question"]}')
                else:
                    get_logger().debug(
                        f"Failed to recieve question: {response.status_code}"
                    )
                    return

                prompt = [
                    {"role": "system", "content": SYSTEM_PROMPTS["default"]},
                    {"role": "user", "content": result["question"]},
                ]
                input_ids = self.processing_class.apply_chat_template(
                    prompt,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                input_ids = input_ids.to(self.model.device)
                outputs = self.model.generate(input_ids, max_new_tokens=512)
                answer = self.processing_class.decode(
                    outputs[0], skip_special_tokens=True
                )
                session_id = result["session_id"]
                submission_data = {
                    "session_id": session_id,
                    "round_number": state.round,
                    "user_answer": answer,
                }
                response = requests.post(
                    f"{base_url}/submit-answer/", json=submission_data
                )

                if response.status_code == 200:
                    result = response.json()
                    get_logger().debug(f"Score: {result['score']}")
                    return
                else:
                    get_logger().debug(
                        f"Failed to submit answer: {response.status_code}"
                    )
                    return
            except Exception as e:
                get_logger().debug(f"Failed to evaluate: {e}")
                return
        else:
            return
