



# 1.baseline
# export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py  --data /home/polina/projects/test/data/ --results /home/polina/projects/test/results --experiment_name baseline --aug --scheduler --learning_rate 0.0001 --epochs 60 --amp --save_ckpt

#  export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --exec_mode evaluate --data /home/polina/projects/test/data/ --results /home/polina/projects/test/results --experiment_name baseline_test --aug --scheduler --amp --ckpt_path /home/polina/projects/test/results/PlaneClassification/230asrq3/checkpoints/best_epoch\=48-acc_\=0.946.ckpt


# 3.loss_weights without reduction='mean' del by 1
# export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py  --data /home/polina/projects/test/data/ --results /home/polina/projects/test/results --experiment_name loss_weights_without_red_mean_norm --aug --loss_weights --scheduler --learning_rate 0.0001 --epochs 60 --amp --save_ckpt

# export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py --exec_mode evaluate --data /home/polina/projects/test/data/ --results /home/polina/projects/test/results --experiment_name loss_weights_test --aug --loss_weights --scheduler --amp --save_ckpt --ckpt_path /home/polina/projects/test/results/PlaneClassification/5y09o4n2/checkpoints/best_*.ckpt

#  3.4.loss_weights without reduction='mean' del by 1 + intensity (NormalizeIntensity)
# export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py  --data /home/polina/projects/test/data/ --results /home/polina/projects/test/results --experiment_name loss_weights_intensity.2 --aug --loss_weights --scheduler --learning_rate 0.0001 --epochs 60 --amp --save_ckpt


#  3.4.5 loss_weights without reduction='mean' del by 1 + intensity (NormalizeIntensity) HistogramNormalize -ужасно 
# export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py  --data /home/polina/projects/test/data/ --results /home/polina/projects/test/results --experiment_name loss_weights_intensity.histogram --aug --loss_weights --scheduler --learning_rate 0.0001 --epochs 60 --amp --save_ckpt

# export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py  --data /home/polina/projects/test/data/ --results /home/polina/projects/test/results --experiment_name loss_weights_intensity.contrast --aug --loss_weights --scheduler --learning_rate 0.0001 --epochs 80 --amp --save_ckpt --ckpt_path /home/polina/projects/test/results/PlaneClassification/3cox8a56/checkpoints/best_*.ckpt


#  3.4.5 loss_weights without reduction='mean' del by 1 + intensity (NormalizeIntensity)+hist+contrast
# export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py  --data /home/polina/projects/test/data/ --results /home/polina/projects/test/results --experiment_name loss_weights_intensity.contrast.hist --aug --loss_weights --scheduler --learning_rate 0.0001 --epochs 80 --amp --save_ckpt 


# export CUDA_VISIBLE_DEVICES=1 && python3 ../main.py  --data /home/polina/projects/test/data/ --results /home/polina/projects/test/results --experiment_name loss_weights_intensity.hist_lr --aug --loss_weights --scheduler --learning_rate 0.0003 --epochs 80 --amp --save_ckpt 

# 1.baseline cross_val GroupShuffleSplit
# export CUDA_VISIBLE_DEVICES=0 && python3 ../main.py  --data /home/polina/projects/test/data/ --results /home/polina/projects/test/results --experiment_name baseline_GroupShuffleSplit_fold_0 --cross_val --nfolds 3 --fold 0 --aug --scheduler --learning_rate 0.0001 --epochs 60 --amp --save_ckpt

# export CUDA_VISIBLE_DEVICES=0 && python3 ../main.py  --data /home/polina/projects/test/data/ --results /home/polina/projects/test/results --experiment_name baseline_GroupShuffleSplit_fold_1 --cross_val --nfolds 3 --fold 1 --aug --scheduler --learning_rate 0.0001 --epochs 60 --amp --save_ckpt

# export CUDA_VISIBLE_DEVICES=0 && python3 ../main.py  --data /home/polina/projects/test/data/ --results /home/polina/projects/test/results --experiment_name baseline_GroupShuffleSplit_fold_2 --cross_val --nfolds 3 --fold 2 --aug --scheduler --learning_rate 0.0001 --epochs 60 --amp --save_ckpt

export CUDA_VISIBLE_DEVICES=0 && python3 ../main.py  --data /home/polina/projects/test/data/ --results /home/polina/projects/test/results --experiment_name resnet --resnet --aug --loss_weights  --scheduler --learning_rate 0.0001 --epochs 60 --amp --save_ckpt