import numpy as np

import os
from PIL import Image

from src.datasets.standard_dataset import DatasetWithSpecifications


class ISICDataset(DatasetWithSpecifications):
    def __init__(self, data_dir: str, **dataset_kwargs):
        """
        :param data_dir: The folder containing dataset
        :param split: train/dev/test string
        """
        patch_not_cancer_path = f"{data_dir}/processed/patch_no_cancer_again/"
        non_patch_not_cancer_path = f"{data_dir}/processed/no_cancer/"
        cancer_path = f"{data_dir}/processed/cancer/"

        self.all_input_fnames, self.all_labels, self.all_mask_fnames, self.input_type = [], [], [], []
        for name in ["non_patch_not_cancer", "patch_not_cancer", "bazooka"]:
            if name == "non_patch_not_cancer":
                _label = 0
                _path = non_patch_not_cancer_path
                _seg_path = None
                _type = 0
            elif name == "patch_not_cancer":
                _label = 0
                _path = patch_not_cancer_path
                _seg_path = f"{data_dir}/segmentation/"
                _type = 1
            else:
                _label = 1
                _path = cancer_path
                _seg_path = None
                _type = 2

            # listdir returns file names in random order
            _fnames = sorted(os.listdir(_path))
            _input_fnames = [f"{_path}/{_fname}" for _fname in _fnames]
            _labels = [_label]*len(_input_fnames)
            _mask_fnames = [None]*len(_input_fnames)
            if _seg_path:
                _mask_fnames = [f"{_seg_path}/{_fname}" for _fname in _fnames]
            self.all_mask_fnames += _mask_fnames
            self.all_labels += _labels
            self.all_input_fnames += _input_fnames
            self.input_type += [_type]*len(_input_fnames)

        # todo: revisit, it sounds a bit unfair to assume we know which images have patches even for val/test.
        #  which is why sticking with evaluation of CDEP
        rng = np.random.default_rng(42)
        train_ln, val_ln = int(0.75 * len(self.all_labels)), int(0.1 * len(self.all_labels))
        _idxs = rng.permutation(np.arange(len(self.all_labels)))
        self.split_dict = {'train': _idxs[:train_ln],
                           'val': _idxs[train_ln: train_ln+val_ln],
                           'test': _idxs[train_ln+val_ln:]}
        if False:
            patch_idxs = np.where(np.array(self.input_type) == 1)[0]
            train_idxs, val_idxs, test_idxs = [self.split_dict[_split] for _split in ['train', 'val', 'test']]
            new_train_idxs, new_val_idxs, new_test_idxs = train_idxs.tolist(), [], []
            for idx in val_idxs:
                if idx in patch_idxs:
                    new_train_idxs.append(idx)
                else:
                    new_val_idxs.append(idx)
            for idx in test_idxs:
                if idx in patch_idxs:
                    new_train_idxs.append(idx)
                else:
                    new_test_idxs.append(idx)
            self.split_dict = {'train': np.array(new_train_idxs), 'val': np.array(new_val_idxs),
                               'test': np.array(new_test_idxs)}

            # npnc = np.where(np.array(self.input_type) == 0)[0]
            # pnc = np.where(np.array(self.input_type) == 1)[0]
            # npc = np.where(np.array(self.input_type) == 2)[0]
            # new_train_idxs, new_val_idxs, new_test_idxs = [], [], []
            # for inp_type, type_split in enumerate([npnc, pnc, npc]):
            #     _idxs = rng.permutation(type_split)
            #     if inp_type == 1:
            #         new_train_idxs += _idxs.tolist()
            #     else:
            #         _l1, _l2 = int(0.75*len(_idxs)), int(0.1*len(_idxs))
            #         new_train_idxs += _idxs.tolist()[:_l1]
            #         new_val_idxs += _idxs.tolist()[_l1:_l1+_l2]
            #         new_test_idxs += _idxs.tolist()[_l1+_l2:]
            # new_train_idxs, new_val_idxs, new_test_idxs = np.array(new_train_idxs), np.array(new_val_idxs), \
            #                                               np.array(new_test_idxs)
            # new_train_idxs = rng.permutation(new_train_idxs)
            # self.split_dict = {'train': new_train_idxs, 'val': new_val_idxs, 'test': new_test_idxs}

    def __len__(self):
        return sum([len(self.split_dict[_k]) for _k in self.split_dict.keys()])

    def __getitem__(self, idx):
        img = Image.open(self.all_input_fnames[idx])
        label = self.all_labels[idx]
        mask = np.zeros(img.size, dtype=np.bool)
        if self.all_mask_fnames[idx]:
            mask = Image.open(self.all_mask_fnames[idx])
            mask = np.asarray(mask)[:, :, 0] > 100

        img = np.array(img)
        group = self.input_type[idx]

        # torch behaves weirdly when changing type from boolean. For instance, bool->long changes the scale of value
        # from 0-1->0-255
        mask = mask.astype(np.float32)
        mask /= max(mask.max(), 1e-5)
        return img, label, mask, group

    def targets(self):
        return np.array(self.all_labels)

    @property
    def num_classes(self) -> int:
        return 2


class ISICGroupedTestDataset(ISICDataset):
    def __init__(self, data_dir, **dataset_kwargs):
        super().__init__(data_dir, **dataset_kwargs)
        test_idxs = self.split_dict["test"]
        test_group_idxs = []
        for grp_name, grp in zip(["ncnp", "ncp", "cnp"], [0, 1, 2]):
            custom_idxs = []
            for ti in test_idxs:
                x, y, m = self[ti]
                if (grp_name == "ncnp") and (m.sum() == 0) and (y == 0):
                    custom_idxs.append(ti)
                elif (grp_name == "ncp") and (m.sum() > 0) and (y == 0):
                    custom_idxs.append(ti)
                elif (grp_name == "cnp") and (y == 1):
                    assert m.sum() == 0
                    custom_idxs.append(ti)
            test_group_idxs.append(custom_idxs)
        self.split_dict["test"] = test_group_idxs
        print("Debug...", list(map(len, test_group_idxs)))

# a data visualisation helper is in notebook/dataset_sanity_check.ipynb


