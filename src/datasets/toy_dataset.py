from src.datasets.standard_dataset import DatasetWithSpecifications
import numpy as np
import torch


class ToyDataset(DatasetWithSpecifications):
    """
    Although I see that the fit gets more vertical if I increase rrr_ap_lamb further (from 100/1000), I am convinced
    that it is not because of RRR but due to some other confounding effect. This is because the rrr_ap_loss is very low
    both in value and visibly even with ap_lamb=100, and yet we seem to see better performance with increasing ap_lamb.
    For comparison, erm has ap_loss between 0.1-0.5 (begins with 1e-4 though), which is sufficiently large. Why is there
    difference in the fit when ap_loss is of the order 1e-6 (ap_loss=100) or 1-8 (ap_loss=1000)?
    Or in other words, for any value of rrr_ap_lamb, there exists a value with which I can separate the groups so that
    I see local invariance but global deviation.
    So, I am sticking with ap_lamb=100 for visual appeal.

    We see the desired effect even with ap_lamb=1000 and when mu, scale, offset, slab = 2, 0.3, 1.4, 20, ap_loss was of
    the order 1e-8
    """
    def __init__(self, data_dir, **dataset_kwargs):
        num_groups = dataset_kwargs.get('num_groups', 3)
        size = dataset_kwargs.get('size', 100)
        rng = np.random.default_rng(0)
        # mu, scale, offset, slab = 2, 0.3, 1.4, 20
        mu, scale, offset, slab = 2, 0.2, 1.2, 20
        OVERALL_SCALE = 1
        mu, scale, offset, slab = mu*OVERALL_SCALE, scale*OVERALL_SCALE, offset*OVERALL_SCALE, slab*OVERALL_SCALE
        all_x, all_y = [], []
        for gi in range(num_groups):
            x_offset, y_offset = (gi % 2)*offset, (gi-1)*slab
            x1 = rng.normal(loc=-mu, scale=scale, size=[size])
            x2 = rng.normal(loc=0, scale=scale, size=[size])
            x3 = rng.normal(loc=mu, scale=scale, size=[size])

            _x = np.concatenate([x1, x2, x3])
            _x = np.stack([_x + x_offset, np.zeros([3*size]) + y_offset], axis=1)
            _y = np.concatenate([np.zeros([size]), np.ones([size]), np.zeros([size])]).astype(np.long)
            all_x.append(_x)
            all_y.append(_y)
        self.all_x, self.all_y = np.concatenate(all_x).astype(np.float32), np.concatenate(all_y).astype(np.long)

        ln = len(self.all_y)
        all_idxs = rng.permutation(ln)
        train_ln, val_ln = int(ln*0.7), int(ln*0.1)
        tr_idxs, val_idxs, te_idxs = all_idxs[:train_ln], all_idxs[train_ln:train_ln+val_ln], all_idxs[train_ln+val_ln:]
        self.split_dict = {
            'train': tr_idxs,
            'val': val_idxs,
            'test': te_idxs
        }

    def __getitem__(self, idx: int) -> (np.ndarray, int, np.ndarray):
        return torch.tensor(self.all_x[idx]), self.all_y[idx], torch.tensor([0, 1.]), self.all_y[idx]

    def __len__(self):
        return len(self.all_y)

    @property
    def num_classes(self) -> int:
        return 2

    def targets(self):
        return self.all_y
