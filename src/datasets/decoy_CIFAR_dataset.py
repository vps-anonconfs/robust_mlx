from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import FashionMNIST
from torchvision.transforms import functional as F

import torch
import numpy as np
from PIL import Image

from src.datasets.standard_dataset import DatasetWithSpecifications


class DecoyCIFAR10(DatasetWithSpecifications):
    def __init__(
            self, data_dir, **dataset_kwargs
    ):
        self.decoy_patch_size = 4  # decoy_patch_size
        self.train_mean = [0.4914, 0.4822, 0.4465]
        self.train_std = [0.2023, 0.1994, 0.2010]
        # self.finder = torch.tensor((243, 7, 2))

        # tr_dat = CIFAR10(data_dir, train=True, download=True)
        # te_dat = CIFAR10(data_dir, train=False, download=False)
        dpath = "/".join(data_dir.split('/')[:-1])
        _ddir = "/".join([dpath, "mnist"])
        tr_dat = MNIST(_ddir, download=True, train=True)
        te_dat = MNIST(_ddir, train=False)
        self.dat = torch.utils.data.ConcatDataset([tr_dat, te_dat])
        self.all_labels = np.concatenate([tr_dat.targets, te_dat.targets])

        tr_idxs, te_idxs = np.arange(len(tr_dat)), np.arange(len(te_dat)) + len(tr_dat)
        rng = np.random.default_rng(42)
        tr_idxs = rng.permutation(tr_idxs)
        train_ln = int(0.9 * len(tr_idxs))
        tr_idxs, val_idxs = tr_idxs[:train_ln], tr_idxs[train_ln:]

        self.split_dict = {'train': tr_idxs, 'val': val_idxs, 'test': te_idxs}
        self.pos = np.random.choice(2, size=[len(self.all_labels), 2])
        dpath = "/".join(data_dir.split('/')[:-1])
        self.mnist_dataset_train = MNIST("/".join([dpath, "mnist"]), train=True, download=True)
        select_idxs = [_idx for idx in range(10) for _idx in np.where(self.mnist_dataset_train.targets == idx)[0][:2]]
        self.mnist_dataset_train_targets = np.array([self.mnist_dataset_train.targets[_].item() for _ in select_idxs])
        self.mnist_dataset_train = torch.utils.data.Subset(self.mnist_dataset_train, select_idxs)
        self.mnist_dataset_test = MNIST("/".join([dpath, "mnist"]), train=False, download=False)
        self.mnist_train_mean = np.stack([self.mnist_dataset_train[_][0] for _ in
                                          range(min(len(self.mnist_dataset_train), 1000))], axis=0).mean()

    def Bern(self, p):
        return np.random.rand() < p

    def augment(self, *args, **kwargs):
        return self.augment2(*args, **kwargs)

    def augment2(self, image, digit, randomize=False, pos=None, index=None, mult=25, all_digits=range(10)):
        img = image.copy()
        img = np.stack([img for _ in range(3)], axis=-1)
        expl = np.zeros_like(img)
        final_img = np.ones_like(img)

        w, h, _ = img.shape
        rng = np.random.default_rng(index)
        if randomize:
            patch_value = np.random.choice(all_digits)*mult  # rng.uniform(0, 9*mult, (1,)).astype('int64')
        else:
            patch_value = digit*mult  # rng.uniform(digit * mult, (digit + 1) * mult, (1,)).astype('int64')

        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(size=[w, h//2])
        left = np.random.choice(2)
        if left == 1:
            final_img[:w//2] = np.array(pil_img)
            final_img[w//2:] = patch_value
            expl[w//2:] = 1
        else:
            final_img[w // 2:] = np.array(pil_img)
            final_img[:w // 2] = patch_value
            expl[:w // 2] = 1
        return final_img, expl[:, :, 0].astype(bool)

    def augment3(self, image, digit, randomize=False, pos=None, index=None, mult=25, all_digits=range(10)):
        img = image.copy()
        expl = np.zeros_like(img)

        fwd = np.arange(self.decoy_patch_size)
        rev = -np.arange(1, self.decoy_patch_size+1)

        rng = np.random.default_rng(index)
        if randomize:
            patch_value = np.random.choice(all_digits)*mult  # rng.uniform(0, 250, (1,)).astype('int64')
        else:
            patch_value = digit*mult  # rng.uniform(digit * mult, (digit + 1) * mult, (1,)).astype('int64')

        if pos is None:
            dir1 = fwd if self.Bern(0.5) else rev
            dir2 = fwd if self.Bern(0.5) else rev
        else:
            dir1, dir2 = [fwd, rev][pos[0]], [fwd, rev][pos[1]]

        for i in dir1:
            for j in dir2:
                for c in [0, 1, 2]:
                    img[i][j][c] = patch_value
                    expl[i][j][c] = 1

        return img, expl[:, :, 0].astype(bool)

    def augment4(self, image, digit, randomize=False, pos=None, mult=25, all_digits=range(10), index=None):
        img = image.copy()
        img = np.stack([img for _ in range(3)], axis=-1)
        expl = np.zeros_like(img)

        patch_size = 6
        start_x, start_y = np.random.choice(29 - patch_size), np.random.choice(29 - patch_size)

        for i in range(patch_size):
            for j in range(patch_size):
                for c in [0, 1, 2]:
                    if randomize:
                        img[start_x + i][start_y + j][c] = 255 - mult * np.random.choice(all_digits)
                    else:
                        img[start_x + i][start_y + j][c] = 255 - mult * digit
                    expl[start_x + i][start_y + j][c] = 1

        return img, expl[:, :, 0].astype(bool)

    def augment5(self, image, digit, randomize=False, pos=None, mult=25, all_digits=range(10), index=None):
        img = image.copy()
        expl = np.zeros_like(img)
        final_img = np.ones_like(img)

        w, h, _ = img.shape
        if randomize:
            _y = np.random.choice(all_digits)
            mnist_idx = np.random.choice(np.where(self.mnist_dataset_test.targets == _y)[0])
            mnist_pil_img = self.mnist_dataset_test[mnist_idx][0].resize(size=[w, h // 2])
            mnist_img = np.array(mnist_pil_img)  #)*self.mnist_train_mean
        else:
            _y = digit
            mnist_idx = np.random.choice(np.where(self.mnist_dataset_train_targets == _y)[0])
            mnist_pil_img = self.mnist_dataset_train[mnist_idx][0].resize(size=[w, h // 2])
            mnist_img = np.array(mnist_pil_img)

        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(size=[w, h // 2])
        left = 0  # np.random.choice(2)
        if left == 1:
            final_img[:w // 2] = np.array(pil_img)
            for c in range(3):
                final_img[w // 2:, :, c] = np.array(mnist_pil_img)
            expl[w // 2:] = 1
        else:
            final_img[w // 2:] = np.array(pil_img)
            for c in range(3):
                final_img[:w // 2, :, c] = mnist_img
            expl[:w // 2] = 1
        return final_img, expl[:, :, 0].astype(bool)

    def augment6(self, image, digit, randomize=False, pos=None, mult=25, all_digits=range(10), index=None):
        if randomize:
            patch_value = np.random.choice(all_digits) * mult
        else:
            patch_value = digit * mult

        padding = 10
        pil_img = Image.fromarray(image)
        pil_img = F.pad(pil_img, padding=padding, fill=patch_value)
        dummy_pil_img = Image.fromarray(np.zeros_like(image))
        dummy_pil_img = F.pad(dummy_pil_img, padding=padding, fill=patch_value)
        img = np.array(pil_img)
        expl = np.ones_like(img)

        expl *= (np.array(dummy_pil_img) == patch_value)
        return img, expl[:, :, 0].astype(bool)

    def augment7(self, image, digit, randomize=False, pos=None, mult=25, all_digits=range(10), index=None):
        image = np.stack([image.copy() for _ in range(3)], axis=-1)
        if randomize:
            patch_value = np.random.choice(all_digits) * mult
        else:
            patch_value = digit * mult

        w, h = 28, 28
        _sh = image.shape
        canvas = np.ones([_sh[0]+10, _sh[1]+10, _sh[2]], dtype=image.dtype)*patch_value
        _x, _y = np.random.choice(10), np.random.choice(10)
        canvas[_x: _x+w, _y: _y+h, :] = image.copy()
        expl = np.ones_like(canvas)
        expl[_x: _x+w, _y: _y+h, :] = 0
        return canvas, expl[:, :, 0].astype(bool)

    def old_augment(self, image, digit, randomize=False, pos=None, mult=25, all_digits=range(10)):
        img = image.copy()
        expl = np.zeros_like(img)

        fwd = [0, 1, 2, 3]
        rev = [-1, -2, -3, -4]
        if pos is None:
            dir1 = fwd if self.Bern(0.5) else rev
            dir2 = fwd if self.Bern(0.5) else rev
        else:
            dir1, dir2 = [fwd, rev][pos[0]], [fwd, rev][pos[1]]

        for i in dir1:
            for j in dir2:
                for c in [0, 1, 2]:
                    if randomize:
                        img[i][j][c] = np.random.choice(256)
                    else:
                        img[i][j][c] = 255 - mult * digit
                    expl[i][j][c] = 1

        return img, expl[:, :, 0].astype(bool)

    def __getitem__(self, index: int):
        img, target = self.dat[index]
        img = np.array(img)
        val_test = (index in self.split_dict['val']) or (index in self.split_dict['test'])
        img, mask = self.augment(img, target, randomize=val_test, pos=self.pos[index], index=index)
        return img, target, mask, target

    def targets(self):
        return self.all_labels

    @property
    def num_classes(self):
        return 10
