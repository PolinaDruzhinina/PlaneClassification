# Multiclass Classification on 2D data

This repository allows users to run a multiclass classification on own datasets. The general training pipline created based on Pytorch Lightning with Weights & Biases frameworks. There are several hard and soft pre-requisites, which include:

```python
torch==1.8.1
torchvision==0.9.1
pandas==1.2.5
numpy==1.21.3
scipy==1.6.3
matplotlib==3.4.1
scikit_learn==0.24.2
monai==0.8.0
tensorboard==2.7.0
pytorch-lightning==1.6.5
torchmetrics==0.11.0
```

#### This repository contains:
1. `main.py` - entry point to the application. Runs training, evaluation;
2. `model.py` - file with model and training pipeline;
3. `data/`  - folder wich contain dataset file;
4. `data/data_module.py` - defines LightningDataModule used by PyTorch Lightning;
5. `notebooks` - folder with jupyter notebook with baseline;
6. `utils.py` - defines utility functions

The main entry-point of the project is through the `main.py` file. 

## Running an experiment
Running an experiment is simple:

```python
python3 ../main.py  --data /path_to_/data/ --results /path_to/results --experiment_name baseline --aug --scheduler --learning_rate 0.0001 --epochs 60 --amp --save_ckpt

```

## Run validation runs to validate your model performance:

```python
python3 ../main.py --exec_mode evaluate --data /path_to/data/ --results /path_to/results --experiment_name baseline_test --aug --scheduler --amp --ckpt_path /path_to_checpoints.ckpt
```

