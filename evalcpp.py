import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="使用 CodeLlama-7b 评估 MBCPP 数据集")
    parser.add_argument("--model_path", type=str, default="codellama/CodeLlama-7b-hf", 
                        help="模型路径或Hugging Face模型ID")
    parser.add_argument("--dataset_path", type=str, default="MBPP/mbcpp", 
                        help="MBCPP数据集路径")
    parser.add_argument("--output_dir", type=str, default="./results", 
                        help="结果输出目录")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--temperature", type=float, default=0.2, help="生成温度")
    parser.add_argument("--num_samples", type=int, default=None, help="评估样本数量，为None则评估全部")
    return parser.parse_args()

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

def format_cpp_prompt(problem):
    """将问题格式化为C++提示"""
    task = problem["text"]
    return f"""编写一个C++函数解决以下问题:
{task}

请仅提供C++代码实现，不要包含解释:
```cpp
"""

def generate_solution(model, tokenizer, prompt, max_length, temperature, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=1,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取生成的部分(去除提示)
    completion = generated_text[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
    
    # 尝试提取生成的代码
    try:
        if "```" in completion:
            code = completion.split("```")[0].strip()
        else:
            code = completion.strip()
        return code
    except:
        return completion.strip()

def evaluate_model(args):
    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, device)
    
    # 加载数据集
    print(f"正在加载数据集: {args.dataset_path}")
    try:
        dataset = load_dataset(args.dataset_path)
        if "test" in dataset:
            eval_set = dataset["test"]
        elif "validation" in dataset:
            eval_set = dataset["validation"]
        else:
            eval_set = dataset["train"]
    except:
        print(f"无法直接加载数据集，尝试从文件加载")
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            eval_set = dataset
    
    # 限制评估样本数量
    if args.num_samples is not None and args.num_samples < len(eval_set):
        eval_set = eval_set[:args.num_samples]
    
    results = []
    
    # 开始评估
    print(f"开始评估，共{len(eval_set)}个样本")
    for i, problem in enumerate(tqdm(eval_set)):
        # 格式化提示
        prompt = format_cpp_prompt(problem)
        
        # 生成解决方案
        try:
            generated_code = generate_solution(
                model, tokenizer, prompt, args.max_length, args.temperature, device
            )
            
            # 保存结果
            result = {
                "problem_id": i,
                "task": problem["text"],
                "generated_code": generated_code,
                "reference_code": problem.get("cpp_solution", "")
            }
            
            results.append(result)
            
            # 定期保存中间结果
            if (i + 1) % 10 == 0:
                with open(os.path.join(args.output_dir, f"intermediate_results_{i+1}.json"), "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
    
    # 保存最终结果
    final_output_file = os.path.join(args.output_dir, "codellama7b_mbcpp_results.json")
    with open(final_output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"评估完成，结果保存至 {final_output_file}")

if __name__ == "__main__":
    args = parse_args()
    evaluate_model(args)