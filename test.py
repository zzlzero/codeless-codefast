import logging
import os
from dataclasses import dataclass
from datetime import datetime
import logging
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import random
import re 
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser
import threading
import time


########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = "google-research-datasets/mbpp"
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None


########################
# Setup logging
########################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

########################
# Reward functions
########################
from lm_eval.tasks.mbpp import MBPP
task = MBPP()
def len_reward_func(completions, **kwargs):
    rewards = []
    max_len = max(len(completion) for completion in completions)
    for completion in completions:

        generation = task.postprocess_generation(completion)
        rewards.append(0.5 - ((len(completion) - len(generation)) / (max_len - len(generation) + 1e-6)))
    return rewards




def main():
    train_dataset = load_dataset("google-research-datasets/mbpp", split="test")
    #####################
    # Prepare and format dataset
    #####################
    def get_prompt(doc):
        prompt = task.get_prompt(doc)
        print(prompt)
        return prompt
    # convert dataset to the prompt
    train_dataset = train_dataset.map(lambda x: {"prompt": get_prompt(x)})

main()