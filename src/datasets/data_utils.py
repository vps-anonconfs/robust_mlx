import numpy as np
from typing import Union, List

from src.datasets.standard_dataset import DatasetWithSpecifications


class DataSubset(DatasetWithSpecifications):
    def __init__(self, dataset: DatasetWithSpecifications, idxs: Union[List, np.ndarray]):
        self.dataset = dataset
        self.idxs = idxs

    def __getitem__(self, item_idx: int) -> (np.ndarray, int, np.ndarray):
        _idx = self.idxs[item_idx]
        return self.dataset[_idx]

    def __len__(self):
        return len(self.idxs)

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes

    def targets(self):
        return self.dataset.targets()[self.idxs]

