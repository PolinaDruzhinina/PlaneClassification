import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import numpy as np
import torch

def set_cuda_devices(args):
    assert args.gpus <= torch.cuda.device_count(), f"Requested {args.gpus} gpus, available {torch.cuda.device_count()}."
    device_list = ",".join([str(i) for i in range(args.gpus)])
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", device_list)

def verify_ckpt_path(args):
    resume_path = os.path.join(args.results, "checkpoints", "last.ckpt")
    if args.resume_training and os.path.exists(resume_path):
        return resume_path
    return args.ckpt_path

def positive_int(value):
    ivalue = int(value)
    assert ivalue > 0, f"Argparse error. Expected positive integer but got {value}"
    return ivalue


def non_negative_int(value):
    ivalue = int(value)
    assert ivalue >= 0, f"Argparse error. Expected non-negative integer but got {value}"
    return ivalue


def float_0_1(value):
    fvalue = float(value)
    assert 0 <= fvalue <= 1, f"Argparse error. Expected float value to be in range (0, 1), but got {value}"
    return fvalue

def get_main_args(strings=None):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg(
        "--exec_mode",
        type=str,
        choices=["train", "evaluate"],
        default="train",
        help="Execution mode to run the model",
    )
    arg("--data", type=str, default="/data", help="Path to data directory")
    arg("--results", type=str, default="/results", help="Path to results directory")
    arg("--experiment_name", type=str, default='baseline0.1', help="Name of experiments ")
    arg("--gpus", type=non_negative_int, default=1, help="Number of gpus")
    arg("--learning_rate", type=float, default=0.0008, help="Learning rate")
    arg("--gradient_clip_val", type=float, default=0, help="Gradient clipping norm value")
    arg("--aug", action="store_true", help="Enable augmentation")
    arg("--loss_weights", action="store_true", help="Enable weights in loss function" )
    arg("--out_channels", type=positive_int, default=4, help="Out channels in model = number classes")
    arg("--amp", action="store_true", help="Enable automatic mixed precision")
    arg("--sync_batchnorm", action="store_true", help="Enable synchronized batchnorm")
    arg("--save_ckpt", action="store_true", help="Enable saving checkpoint")
    arg("--seed", type=non_negative_int, default=42, help="Random seed")
    arg("--ckpt_path", type=str, default=None, help="Path to checkpoint")
    arg("--cross_val", action="store_true", help="Enable cross validation" )
    arg("--resnet", action="store_true", help="Enable resnet architecture" )
    arg("--auc", action="store_true", help="Calculate metric auc" )
    arg("--nfolds", type=positive_int, default=5, help="Number of cross-validation folds")
    arg("--fold", type=non_negative_int, default=0, help="Fold number")
    arg("--patience", type=positive_int, default=10, help="Early stopping patience")
    arg("--batch_size", type=positive_int, default=128, help="Batch size")
    arg("--val_batch_size", type=positive_int, default=64, help="Validation batch size")
    arg("--profile", action="store_true", help="Run dlprof profiling")
    arg("--momentum", type=float, default=0.99, help="Momentum factor")
    arg("--weight_decay", type=float, default=0.0001, help="Weight decay (L2 penalty)")
    arg("--save_preds", action="store_true", help="Enable prediction saving")
    arg("--resume_training", action="store_true", help="Resume training from the last checkpoint")
    arg("--num_workers", type=non_negative_int, default=10, help="Number of subprocesses to use for data loading")
    arg("--epochs", type=non_negative_int, default=60, help="Number of training epochs")
    arg("--warmup", type=non_negative_int, default=5, help="Warmup iterations before collecting statistics")
    arg("--norm", type=str, choices=["instance", "batch", "group"], default="instance", help="Normalization layer")
    arg(
        "--scheduler",
        action="store_true",
        help="Enable cosine rate scheduler with warmup",
    )
    arg(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "adam"],
        help="Optimizer",
    )
    if strings is not None:
        arg(
            "strings",
            metavar="STRING",
            nargs="*",
            help="String for searching",
        )
        args = parser.parse_args(strings.split())
    else:
        args = parser.parse_args()
    return args
