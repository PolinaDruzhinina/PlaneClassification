import ctypes
import os

import nvidia_dlprof_pytorch_nvtx
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, early_stopping
from pytorch_lightning.loggers import WandbLogger
from data.data_module import DataModule
from utils import get_main_args, set_cuda_devices, verify_ckpt_path
from model import NetModule

if __name__ == "__main__":
    args = get_main_args()


    set_cuda_devices(args)
    seed_everything(args.seed)
    data_module = DataModule(args)
    data_module.prepare_data()
    data_module.setup()
    weights = data_module.get_weights()
    ckpt_path = verify_ckpt_path(args)

    callbacks = None
    model_ckpt = None
    wandb_logger = WandbLogger(name=args.experiment_name, project='PlaneClassification')
    if args.exec_mode == "train":
        model = NetModule(args,weights)
        early_stopping = EarlyStopping(monitor="val_acc", patience=args.patience, verbose=True, mode="max")
        callbacks = [early_stopping]
        if args.save_ckpt:
            model_ckpt = ModelCheckpoint(
                filename="best_{epoch}-{acc:.2f}", monitor="val_acc", mode="max", save_last=True, save_top_k =1)
            callbacks.append(model_ckpt)
    else:  # Evaluation 
        if ckpt_path is not None:
            model = NetModule.load_from_checkpoint(ckpt_path)
        else:
            model = NetModule(args,weights)

    trainer = Trainer(
        logger=wandb_logger,
        gpus=args.gpus,
        precision=16 if args.amp else 32,
        benchmark=True,
        deterministic=False,
        min_epochs=args.epochs,
        max_epochs=args.epochs,
        sync_batchnorm=args.sync_batchnorm,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        default_root_dir=args.results,
        resume_from_checkpoint=ckpt_path,
        accelerator="ddp" if args.gpus > 1 else None,
        log_every_n_steps=10,
    )

    if args.exec_mode == "train":
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
    elif args.exec_mode == "evaluate":
        ckpt_name = "_".join(args.ckpt_path.split("/")[-1].split(".")[:-1])    
        model.args = args
        trainer.test(model, datamodule=data_module)
