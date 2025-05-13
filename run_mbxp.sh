CUDA_VISIBLE_DEVICES=4,5,6,7
accelerate launch --num_processes 3 --config_file ./deepspeed_zero3.yaml --main_process_port 29501 ./run_grpo_mbxp.py --config ./codellama-7b-grpo_mbxp.yaml 2>&1 | tee run_grpo_mbxp_all.log
