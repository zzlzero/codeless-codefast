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
from lm_eval.tasks.mbxp import MBXP
from lm_eval.tasks.custom_metrics.multiple_metrics.containerized_eval import eval_string_script
language = 'cpp'
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


def grpo_function(
    model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    #########################
    # Log parameters
    #########################
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Load tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # 如果eos_token也是None，添加一个专门的pad_token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    ###############
    # Load datasets
    ###############
    # Load dataset from Hugging Face Hub

    train_dataset = task.get_dataset(use_train=True)

    test_dataset = task.get_dataset(use_train=False)

    #########################
    # Instantiate DPO trainer
    #########################

    trainer = GRPOTrainer(
      model=model_args.model_name_or_path,
      reward_funcs=[len_reward_func, correct_code_reward_func],
      processing_class=tokenizer,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=test_dataset,
      peft_config=get_peft_config(model_args),
    )


    ###############
    # Training loop
    ###############
    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Train the model
    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for {training_args.num_train_epochs} epochs***'
    )
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################

    logger.info("*** Save model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    training_args.distributed_state.wait_for_everyone()  # wait for all processes to load

    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl","grpo", "tutorial"]})
    # push to hub if needed
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()