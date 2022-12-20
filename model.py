import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision
from monai.networks.nets import DenseNet121
from monai.metrics import ROCAUCMetric
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from monai.data import decollate_batch
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose
)
import pytorch_lightning as pl

def create_model(classes):
    model = torchvision.models.resnet18(pretrained=False, num_classes=classes)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

class NetModule(pl.LightningModule):
    def __init__(self, args,class_weights):
        """
        Inputs:
           args - arguments
           class_weights - weights per each data class for weighted loss 
        """
        super().__init__()
        self.args = args
        self.save_hyperparameters()
        if self.args.resnet:
            self.model = create_model(self.args.out_channels)
        else: 
            self.model = DenseNet121(spatial_dims=2, in_channels=1,
                        out_channels=self.args.out_channels)
        if self.args.loss_weights:
            self.loss_module = nn.CrossEntropyLoss(class_weights)
        else:
            self.loss_module = nn.CrossEntropyLoss() 
        
        self.learning_rate = args.learning_rate
        self.y_pred_trans = Compose([Activations(softmax=True)])
        self.y_trans = Compose([AsDiscrete(to_onehot=self.args.out_channels)])
        self.auc_metric = ROCAUCMetric()
        
        
    def forward(self, imgs):
        return self.model(imgs)
    
    def configure_optimizers(self):
        optimizer = {
            "sgd": SGD(self.parameters(), lr=self.learning_rate, momentum=self.args.momentum),
            "adam": Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay),
        }[self.args.optimizer.lower()]

        if self.args.scheduler:
            scheduler = {
                "scheduler": WarmupCosineSchedule(
                    optimizer=optimizer,
                    warmup_steps=250,
                    t_total=self.args.epochs * len(self.trainer.datamodule.train_dataloader()),
                ),
                "interval": "step",
                "frequency": 1,
            }
            return {"optimizer": optimizer, "monitor": "val_loss", "lr_scheduler": scheduler}
        return {"optimizer": optimizer, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss 

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        prediction = self.model(imgs)
        loss = self.loss_module(prediction, labels)
        acc = (labels == prediction.argmax(dim=-1)).float().mean()
        y_onehot = [self.y_trans(i) for i in decollate_batch(labels, detach=False)]
        y_pred_act = [self.y_pred_trans(i) for i in decollate_batch(prediction)]
        self.auc_metric(y_pred_act, y_onehot)
        result = self.auc_metric.aggregate()
        self.auc_metric.reset()
        del y_pred_act, y_onehot
        
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)
        self.log("val_auc", result)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        prediction = self.model(imgs)
        acc = (labels == prediction.argmax(dim=-1)).float().mean()
        
        y_onehot = [self.y_trans(i) for i in decollate_batch(labels, detach=False)]
        y_pred_act = [self.y_pred_trans(i) for i in decollate_batch(prediction)]
        self.auc_metric(y_pred_act, y_onehot)
        result = self.auc_metric.aggregate()
        self.auc_metric.reset()
        del y_pred_act, y_onehot
        
        self.log("test_acc", acc)
        self.log("test_auc", result)