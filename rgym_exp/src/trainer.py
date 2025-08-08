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
        # Handle DataParallel wrapped models
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.generate(*args, **kwargs)
        else:
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
