import argparse
import csv
import json
import os
import subprocess
import tempfile
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval.tasks.mbpp import MBPP


def parse_gammas(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def gamma_to_suffix(gamma: float) -> str:
    return str(gamma).replace("-", "neg").replace(".", "p")


def run_python_tests(generation: str, test_list: List[str], timeout: int) -> bool:
    reference = "\n".join(test_list)
    test_program = generation + "\n" + reference

    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(test_program.encode("utf-8"))

    try:
        result = subprocess.run(
            ["python", temp_filename],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(temp_filename)
        except OSError:
            pass


def evaluate_one_model(
    model_path: str,
    dataset_name: str,
    max_new_tokens: int,
    timeout: int,
    max_samples: int,
):
    task = MBPP()
    dataset = load_dataset(dataset_name, split="test")
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    pass_count = 0
    lengths = []

    for doc in dataset:
        prompt = task.get_prompt(doc)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generation = task.postprocess_generation(generated_text, prompt)
        completion = generated_text
        length_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
        lengths.append(length_tokens)

        if run_python_tests(generation, doc["test_list"], timeout):
            pass_count += 1

    total = len(dataset)
    pass_at_1 = pass_count / total if total else 0.0
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    return pass_at_1, avg_len, total


def main():
    parser = argparse.ArgumentParser(description="Gamma sensitivity evaluation for MBPP")
    parser.add_argument("--gammas", type=str, default="0.1,0.5,1.0,2.0")
    parser.add_argument("--model_dir_template", type=str, default="runs/CodeLlama-7b-hf-gamma-{gamma}")
    parser.add_argument("--dataset", type=str, default="google-research-datasets/mbpp")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means full test set")
    parser.add_argument("--output_json", type=str, default="runs/gamma_sensitivity_results.json")
    parser.add_argument("--output_csv", type=str, default="runs/gamma_sensitivity_results.csv")
    parser.add_argument("--output_md", type=str, default="runs/gamma_sensitivity_results.md")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    rows = []
    for gamma in parse_gammas(args.gammas):
        model_path = args.model_dir_template.format(gamma=gamma_to_suffix(gamma))
        if not os.path.isdir(model_path):
            print(f"[WARN] Skip gamma={gamma}, model path not found: {model_path}")
            continue

        print(f"[INFO] Evaluating gamma={gamma} from {model_path}")
        pass_at_1, avg_len, total = evaluate_one_model(
            model_path=model_path,
            dataset_name=args.dataset,
            max_new_tokens=args.max_new_tokens,
            timeout=args.timeout,
            max_samples=args.max_samples,
        )
        row = {
            "gamma": gamma,
            "pass_at_1": round(pass_at_1, 6),
            "avg_generation_length": round(avg_len, 2),
            "num_samples": total,
        }
        rows.append(row)

    rows = sorted(rows, key=lambda x: x["gamma"])

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["gamma", "pass_at_1", "avg_generation_length", "num_samples"])
        writer.writeheader()
        writer.writerows(rows)

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("| gamma | Pass@1 | Avg Generation Length (tokens) | #Samples |\n")
        f.write("|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['gamma']} | {row['pass_at_1']:.4f} | {row['avg_generation_length']:.2f} | {row['num_samples']} |\\n"
            )

    print(f"[INFO] Saved: {args.output_json}")
    print(f"[INFO] Saved: {args.output_csv}")
    print(f"[INFO] Saved: {args.output_md}")


if __name__ == "__main__":
    main()
