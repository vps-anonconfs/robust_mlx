from src.datasets.decoy_CIFAR_dataset import DecoyCIFAR10
from src.datasets.isic_dataset import ISICDataset, ISICGroupedTestDataset
from src.datasets.cmnist_dataset import CMNISTDataset
from src.datasets.toy_dataset import ToyDataset
from src.datasets.plant_rgb_dataset import PlantRGBDataset

import os


def get_dataset(dataset_name: str, data_dir: str, **dataset_kwargs):
    dataset_name = dataset_name.lower()
    if dataset_name == "isic":
        return ISICDataset(os.path.join(data_dir, "isic"), **dataset_kwargs)
    elif dataset_name == "isic_grouped_test":
        return ISICGroupedTestDataset(os.path.join(data_dir, "isic"), **dataset_kwargs)
    elif dataset_name == 'cmnist':
        return CMNISTDataset(os.path.join(data_dir, "cmnist"), **dataset_kwargs)
    elif dataset_name == 'plant':
        return PlantRGBDataset(os.path.join(data_dir, "plant"), **dataset_kwargs)
    elif dataset_name == 'decoy_cifar10':
        return DecoyCIFAR10(os.path.join(data_dir, "cifar10"), **dataset_kwargs)
    elif dataset_name == 'toy':
        return ToyDataset(None, **dataset_kwargs)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
