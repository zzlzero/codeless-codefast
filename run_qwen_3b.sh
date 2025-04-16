CUDA_VISIBLE_DEVICES=3,4,6,7
accelerate launch --num_processes 3 --config_file ./deepspeed_zero3.yaml --main_process_port 29501 ./run_grpo.py --config ./qwen-3b-grpo.yaml 2>&1 | tee run_qwen_grpo.log
