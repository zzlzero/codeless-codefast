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
from lm_eval.tasks.mbxp import MBXP
from lm_eval.tasks.mbpp import MBPP

from lm_eval.tasks.custom_metrics.multiple_metrics.containerized_eval import eval_string_script
import json
language = 'go'
task = MBXP(language=language)

########################
# Custom dataclasses
########################
@dataclass
class ScriptArguments:
    dataset_id_or_path: str = ""
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None



########################
# Reward functions
########################

def len_reward_func(prompts, completions, **kwargs):
    rewards = []
    max_len = max(len(completion) for completion in completions)
    for prompt, completion in zip(prompts, completions):
        generation = prompt + ' ' + completion
        generation = task.postprocess_generation(generation, prompt)
        rewards.append(0.5 - ((len(completion) - len(generation) + len(prompt) + 1) / (max_len - len(generation) + len(prompt) + 1 + 1e-6)))
    return rewards


def correct_code_reward_func(prompts, completions, test, **kwargs):
    rewards = []
    
    for prompt, reference, completion in zip(prompts, test, completions):
        generation = prompt + ' ' + completion
        generation = task.postprocess_generation(generation, prompt)
        # print(generation)
        test_program = generation + "\n" + reference
        
        result = eval_string_script(language, test_program)
        # print(result['stderr'])
        if result["status"] == "OK":

            # 返回码为0表示测试通过
            rewards.append(1.0)
            
            # 记录成功样本
            if torch.rand(1).item() < 0.10:  # 10% 的概率记录成功样本
                os.makedirs("completion_samples_mbxp", exist_ok=True)
                log_file = os.path.join("completion_samples_mbxp", "success_code_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n==============\n")
                    f.write(f"Prompt:\n{prompt}\n\nGeneration:\n{generation}\n\nTest:\n{reference}\n")
        else:
            # 测试失败
            print(f"Test failed with error: {result['stderr']}")
            rewards.append(0.0)        
        print(f"Reward: {rewards[-1]}")
    
    return rewards


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


###############
# Load datasets
###############
# Load dataset from Hugging Face Hub

mbgp = MBXP(language="go")
# mbjsp = MBXP(language="javascript")
# mbcpp = MBXP(language="cpp")
# mbgp_dataset = mbgp.get_dataset(use_train=True)
# mbcpp_dataset = mbcpp.get_dataset(use_train=True)
# mbjsp_dataset = mbjsp.get_dataset(use_train=True)
# print(len(mbgp_dataset), len(mbcpp_dataset), len(mbjsp_dataset))
# print(mbgp_dataset[0])
# print(mbcpp_dataset[0])
# print(mbjsp_dataset[0])

from execution import check_correctness_go

with open("/data/zzl/CodeFast/results/codellama_7b_mbxp_go_codefast/generations.json", "r") as f:
    generations = json.load(f)
print("Loaded generations.json, length:", len(generations))
mbgp_dataset = mbgp.get_dataset(use_train=False)

problem = mbgp_dataset
print(problem[0])
ref = mbgp_dataset[0]['test']
test_program = generations[0][0] + "\n" + ref
print(test_program)
# result = eval_string_script(language, test_program)
# print(result['stderr'])

for i in range(10):
    ref = mbgp_dataset[i]['test']

    result = check_correctness_go(problem=problem[i], completion=generations[i][0], timeout=30)
    # test_program = generations[i][0] + "\n" + ref
    # result = eval_string_script(language, test_program)
    print(f"Test {i} result:")
    ###############
    # Print the result
    ###############
    # if result["status"] == "OK":
    #     print("Test passed")
    # else:
    #     print("Test failed")
    #     print(test_program)
    # print(result["stdout"])
    # print(result["stderr"])
    # print(result["exit_code"])

    print(result)
# print(result["status"])
# 