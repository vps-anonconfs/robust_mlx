from src.datasets.standard_dataset import DatasetWithSpecifications
from torchvision.datasets import MNIST
from PIL import Image
import torch

import torchvision.datasets as datasets
import numpy as np
import torch.utils.data as utils
from colour import Color
import os
from os.path import join as oj


class CMNISTDataset(DatasetWithSpecifications):
    def __init__(self, data_dir, **dataset_kwargs):
        rng = np.random.default_rng()
        red = Color("red")
        colors = list(red.range_to(Color("purple"), 10))
        self.colors = [np.asarray(x.get_rgb()) for x in colors]

        mnist_trainset = datasets.MNIST(root=data_dir, train=True, download=True, transform=None)
        num_train = int(len(mnist_trainset) * .9)
        _idxs = rng.permutation(np.arange(len(mnist_trainset)))
        train_dat, val_dat = torch.utils.data.Subset(mnist_trainset, _idxs[:num_train]), \
            torch.utils.data.Subset(mnist_trainset, _idxs[num_train:])
        test_dat = datasets.MNIST(root=data_dir, train=False, download=True, transform=None)

        self.dat = torch.utils.data.ConcatDataset([train_dat, val_dat, test_dat])
        train_ln, val_ln = len(train_dat), len(val_dat)
        self.split_dict = {'train': np.arange(train_ln), 'val': np.arange(train_ln, train_ln+val_ln),
                           'test': np.arange(train_ln+val_ln, len(self.dat))}
        self.flip_colors = [0 for _ in range(train_ln)] + [1 for _ in range(len(self.dat)-train_ln)]

        # two kinds of mask: relirrel (of cdep) or irrel (of IBP-Ex)
        # IBP requires a mask that just perturbs the irrelevant features whereas cdep requires a mask that identifies
        # relevant-irrelevant regions of a mask.
        self.mask_type = dataset_kwargs['mask_type']
        # this is the weird mask defined by CDEP
        # https://github.com/laura-rieger/deep-explanation-penalization/blob/master/mnist/ColorMNIST/train.py#L107
        # we continue to use this because it is non-trivial to define relevant and irrelevant pixel regions
        self.blobs = np.zeros((28 * 28, 28, 28), dtype=np.float32)
        for i in range(28):
            for j in range(28):
                # self.blobs[i * 28 + j, i, j] = 1
                for _ in range(10):
                    self.blobs[i*28 + j, np.random.choice(28), np.random.choice(28)] = 1.
        # mask for IBP
        x, _ = self.dat[0]
        self.masks = []
        for _i in range(3):
            _mask = -0.25*np.ones_like(np.array(x.convert('RGB')))
            _mask[:, :, _i] = 1.
            self.masks.append(_mask.astype(np.float32))

        self.all_labels = np.array([y for x, y in train_dat])

    def __getitem__(self, idx: int) -> (np.ndarray, int, np.ndarray):
        x, y = self.dat[idx]
        g = 0
        # w x h x c
        x = x.convert("RGB")
        _color = y if not self.flip_colors[idx] else (9-y)
        x = x * self.colors[_color][None, None, :]
        x = np.array(x).astype(np.uint8)
        if self.mask_type.lower().startswith('irrel'):
            _mi = np.random.choice(len(self.masks))
            m = self.masks[_mi]
        else:
            m = self.blobs[np.random.choice(len(self.blobs))]
        return x, y, m, g

    def __len__(self):
        return len(self.dat)

    @property
    def num_classes(self) -> int:
        return 10

    def targets(self):
        return self.all_labels


if __name__ == '__main__':
    dat = CMNISTDataset(os.path.expanduser("~/data/cmnist"), **{'mask_type': 'irrel'})

