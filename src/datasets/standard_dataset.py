import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetWithSpecifications(Dataset):
    def __getitem__(self, item: int) -> (np.ndarray, int, np.ndarray, int):
        """
        Returns the input, label, mask, group. The input and mask must be of the same size and the mask must be boolean.
        The true values of the mask must identify sensitive or nuisance pixels/words/features of the input.
        """
        raise NotImplementedError()

    @property
    def num_classes(self) -> int:
        raise NotImplementedError()

    def targets(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
