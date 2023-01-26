from typing import Sequence

import pytorch_lightning as pl
from torchvision import transforms
import numpy as np

import torch
from torch.utils.data import DataLoader

from src.datasets import get_dataset
from src.datasets.data_utils import DataSubset


class DataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = kwargs['dataset']

        dataset = get_dataset(self.dataset_name, kwargs['data_dir'], **kwargs['dataset_kwargs'])

        if hasattr(dataset, 'train_mean'):
            train_mean, train_std = dataset.train_mean, dataset.train_std
            print(f"Using dataset stats: {train_mean, train_std}")
        else:
            train_mean, train_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])
        rng = np.random.default_rng(kwargs["data_seed"])
        train_idxs = dataset.split_dict["train"]
        if kwargs["data_frac"] and (kwargs["data_frac"] >= 0):
            train_idxs = rng.choice(train_idxs, kwargs["data_frac"])
        self.train_dataset = DataSubset(dataset, train_idxs)
        self.val_dataset = DataSubset(dataset, dataset.split_dict["val"])
        if isinstance(dataset.split_dict["test"], Sequence):
            self.test_dataset = [DataSubset(dataset, _te) for _te in dataset.split_dict["test"]]
        else:
            self.test_dataset = DataSubset(dataset, dataset.split_dict["test"])
        self.num_classes = dataset.num_classes

    def get_class_weights(self):
        train_labels = self.train_dataset.targets()
        labels, counts = np.unique(train_labels, return_counts=True)
        class_wts = ((counts.sum() - counts) / counts.sum()) * (1 / (self.num_classes - 1))
        return class_wts

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size_train,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.my_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            collate_fn=self.my_collate,
        )

    def test_dataloader(self):
        if isinstance(self.test_dataset, Sequence):
            return [DataLoader(
                ds,
                batch_size=self.hparams.batch_size_test,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                collate_fn=self.my_collate,
            ) for ds in self.test_dataset]
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.hparams.batch_size_test,
                shuffle=False,
                num_workers=self.hparams.num_workers,
                collate_fn=self.my_collate,
            )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size_test,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def my_collate(self, batch):
        # just pass on if it is simple data
        if len(batch[0][0].shape) == 1:
            data = torch.stack([item[0] for item in batch], dim=0)
            target = torch.LongTensor([item[1] for item in batch])
            group = torch.LongTensor([item[-1] for item in batch])
            m = torch.stack([item[2] for item in batch], dim=0)
        else:
            # B x 3 x w x h
            data = torch.stack([self.input_transform(item[0]) for item in batch], dim=0)
            # check the mask shape
            if len(batch[0][2].shape) == 2:
                # B x 1 x w x h
                m = torch.stack([torch.unsqueeze(torch.from_numpy(item[2]), dim=0) for item in batch], dim=0)
            elif len(batch[0][2].shape) == 3:
                m = torch.stack([torch.permute(torch.from_numpy(item[2]), (2, 0, 1)) for item in batch], dim=0)
            else:
                raise ValueError("Unrecognised shape of masks: ", batch[0][2].shape)
            target = [item[1] for item in batch]
            target = torch.LongTensor(target)
            group = torch.LongTensor([item[-1] for item in batch])
        return [data, target, m, group]


