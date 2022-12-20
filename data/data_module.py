import glob
import os
import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from monai.data import decollate_batch, DataLoader
from monai.transforms import (
    AsDiscrete,
    Compose,
    RandGaussianNoised,
    RandAdjustContrast,
    RandGaussianSmooth,
    NormalizeIntensity,
    AddChannel,
    ToTensor
)
from monai.utils import set_determinism
from sklearn.model_selection import train_test_split,StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import h5py


class PlaneDataset(Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


class DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_path = self.args.data
        self.l_t = LabelEncoder()
        self.train_transforms = None
        self.val_transforms = None
        if self.args.aug:
            self.train_transforms = Compose([ AddChannel(),
                                             NormalizeIntensity(),
#                                              RandGaussianNoised(prob=0.5),
#                                              RandAdjustContrast(prob=0.5),
#                                              RandGaussianSmooth(prob=0.5) 
                                             ToTensor()
                                            ])
            self.val_transforms = Compose([AddChannel(),
                                           NormalizeIntensity(),
                                           ToTensor()])

    def get_weights(self):
        return 1/(torch.Tensor([len(self.y_train)/c for c in pd.Series(np.bincount(self.y_train)).sort_index().values]))
            
    def setup(self, stage=None):
        datah5py = h5py.File(os.path.join(self.data_path, 'plane_classification'))
        img, plane_id, sub_id, y = [datah5py[k] for k in datah5py.keys()] 
        y_label_encoding = self.l_t.fit_transform(y)
        
        if self.args.cross_val:
            kfold = self.get_kfold_splitter(self.args.nfolds)
            train_idx, val_idx = list(kfold.split(np.array(img), np.array(y_label_encoding), np.array(sub_id)))[self.args.fold] 
            X_train, X_val, = np.array(img)[train_idx], np.array(img)[val_idx]
            self.y_train, y_val = np.array(y_label_encoding)[train_idx], np.array(y_label_encoding)[val_idx]
        else:
            
            X_train, X_test, self.y_train, y_test = train_test_split(np.array(img), np.array(y_label_encoding), 
                                                                train_size=0.7, 
                                                                random_state=self.args.seed,
                                                                stratify=np.array(y_label_encoding))
            X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, 
                                                                train_size=0.5, 
                                                                random_state=self.args.seed,
                                                                stratify=y_test)
            self.test_ds = PlaneDataset(X_test, y_test, self.val_transforms)
            
        self.train_ds = PlaneDataset(X_train, self.y_train, self.train_transforms)
        self.val_ds = PlaneDataset(X_val, y_val, self.val_transforms)
        
        print(f"Dataset split: \n {np.bincount(self.y_train)} training, {np.bincount(y_val)} validation, {np.bincount(y_val)} test examples")

    def train_dataloader(self):
        return  DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=10)

    def val_dataloader(self):
        return  DataLoader(self.val_ds, batch_size=self.args.val_batch_size, num_workers=10)

    def test_dataloader(self):
        if self.args.cross_val:
            return DataLoader(self.val_ds, batch_size=self.args.val_batch_size, num_workers=10)
        return  DataLoader(self.test_ds, batch_size=self.args.val_batch_size, num_workers=10)

    def get_kfold_splitter(self, nfolds):
        return StratifiedGroupKFold(n_splits=nfolds)
