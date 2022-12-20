# Example to run training
export CUDA_VISIBLE_DEVICES=0 && python3 ../main.py  --data /path_to_/data/ --results /path_to/results --experiment_name baseline --aug --scheduler --learning_rate 0.0001 --epochs 60 --amp --save_ckpt

# Example to run validation
 export CUDA_VISIBLE_DEVICES=0 && python3 ../main.py --exec_mode evaluate --data /path_to/data/ --results /path_to/results --experiment_name baseline_test --aug --scheduler --amp --ckpt_path /path_to_checpoints.ckpt