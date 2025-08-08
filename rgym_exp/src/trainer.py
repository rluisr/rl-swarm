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
                
                # Try various methods to extract input_ids
                input_ids = None
                
                # Method 1: Direct attribute access
                if hasattr(first_arg, 'input_ids'):
                    print("[DEBUG] Using direct attribute access")
                    input_ids = first_arg.input_ids
                
                # Method 2: to_dict() method
                elif hasattr(first_arg, 'to_dict'):
                    print("[DEBUG] Using to_dict() method")
                    try:
                        data_dict = first_arg.to_dict()
                        if 'input_ids' in data_dict:
                            input_ids = data_dict['input_ids']
                    except Exception as e:
                        print(f"[DEBUG] to_dict() failed: {e}")
                
                # Method 3: __getitem__ access
                elif hasattr(first_arg, '__getitem__') and hasattr(first_arg, '__len__'):
                    print("[DEBUG] Using __getitem__ method")
                    try:
                        if len(first_arg) > 0:
                            item = first_arg[0]
                            if isinstance(item, dict) and 'input_ids' in item:
                                input_ids = item['input_ids']
                            elif hasattr(item, 'input_ids'):
                                input_ids = item.input_ids
                    except Exception as e:
                        print(f"[DEBUG] __getitem__ failed: {e}")
                
                # Method 4: Try to access as a dictionary directly
                if input_ids is None:
                    try:
                        if 'input_ids' in first_arg:
                            print("[DEBUG] Using dictionary access")
                            input_ids = first_arg['input_ids']
                    except:
                        pass
                
                # Convert to tensor if we found input_ids
                if input_ids is not None:
                    if not isinstance(input_ids, torch.Tensor):
                        print(f"[DEBUG] Converting to tensor from type: {type(input_ids)}")
                        try:
                            input_ids = torch.tensor(input_ids)
                        except Exception as e:
                            print(f"[DEBUG] Tensor conversion failed: {e}")
                            # Try to convert to list first, then to tensor
                            try:
                                if hasattr(input_ids, 'tolist'):
                                    input_ids = torch.tensor(input_ids.tolist())
                                else:
                                    input_ids = torch.tensor(list(input_ids))
                            except Exception as e2:
                                print(f"[DEBUG] Alternative tensor conversion failed: {e2}")
                    
                    # Replace the first argument
                    if input_ids is not None:
                        print(f"[DEBUG] Successfully processed input_ids, shape: {input_ids.shape if hasattr(input_ids, 'shape') else 'unknown'}")
                        args = (input_ids,) + args[1:]
                else:
                    print("[DEBUG] Could not extract input_ids from Dataset")
            
            # If it's already a tensor, pass through
            elif isinstance(first_arg, torch.Tensor):
                print(f"[DEBUG] First arg is already a tensor")
            
            # Otherwise, try to convert to tensor
            elif not isinstance(first_arg, torch.Tensor):
                print(f"[DEBUG] Attempting to convert {type(first_arg)} to tensor")
                try:
                    tensor_arg = torch.tensor(first_arg)
                    args = (tensor_arg,) + args[1:]
                except Exception as e:
                    print(f"[DEBUG] Tensor conversion failed: {e}, passing through as-is")
        
        # Handle DataParallel wrapped models
        if isinstance(self.model, nn.DataParallel):
            print("[DEBUG] Using DataParallel module.generate")
            return self.model.module.generate(*args, **kwargs)
        else:
            print("[DEBUG] Using direct model.generate")
            return self.model.generate(*args, **kwargs)

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
