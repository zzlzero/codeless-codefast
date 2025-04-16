# import eval_string_script
from lm_eval.tasks.custom_metrics.multiple_metrics.containerized_eval import eval_string_script
import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import torch
from lm_eval.tasks.mbxp import MBXP

def generate_solution(model, tokenizer, prompt, max_length, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            do_sample=False,
            num_return_sequences=1,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
def load_model_and_tokenizer(model_path, device="cuda"):
    print(f"正在加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 确保tokenizer能正确处理代码
    tokenizer.pad_token = tokenizer.eos_token
    
    # 根据设备选择数据类型
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    
    model.eval()
    return model, tokenizer
# 确定设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
    
# 加载模型和tokenizer
model, tokenizer = load_model_and_tokenizer('codellama/CodeLlama-7b-hf', device)
language = 'cpp'
task = MBXP(language=language)

dataset = task.get_dataset()
print(dataset[0])
for example in dataset:
    prompt = task.get_prompt(example)
    output = generate_solution(model, tokenizer, prompt, 500, device)
    
    generation = task.postprocess_generation(output, prompt)
    

    reference = task.get_reference(example)

    test_program = generation + "\n" + reference

    
    result = eval_string_script(language, test_program)
    print(result['stderr'])
    if result["status"] == "OK":
        print("测试通过")

        print(generation)
        