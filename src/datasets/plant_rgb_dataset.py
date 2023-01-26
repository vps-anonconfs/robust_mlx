import numpy as np
from PIL import Image, ImageEnhance

from src.datasets.standard_dataset import DatasetWithSpecifications
import pickle

dai_dict = dict({
    'Z': {
        'dai_offset': 9,
        1: {
            1: -1, 2: -1, 3: -1,
            4: 9, 5: 9, 6: 9, 7: 9, 8: 9,
            9: 14, 10: 14, 11: 14, 12: 14, 13: 14,
            14: 19, 15: 19, 16: 19, 17: 19, 18: 19,
        },
    },
})

# mapping the labels of incomplete dataset
dai_incom_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 7: 6, 8: 7, 9: 8, 12: 9, 13: 10, 14: 11}
dai_incom_dict_inv = {value: key for key, value in dai_incom_dict.items()}

for dai in range(2, 6):
    dai_dict['Z'][dai] = dai_dict['Z'][1].copy()
    for i in dai_dict['Z'][dai].keys():
        if type(i) == int and i >= 4:
            dai_dict['Z'][dai][i] = dai_dict['Z'][dai][i] + dai - 1


def get_dai_label(sample_id):
    """
    get day after incubation given a string of the sample ID.
    sample_id e.g. '1,Z12,...'
    """
    sample_id = sample_id.split(",")
    # sample_id e.g. '1,Z12,...'
    day = sample_id[0]
    plant_type = sample_id[1][0]
    sample_num = sample_id[1][1:]
    label = dai_dict[plant_type][int(day)][int(sample_num)]
    if label == -1:
        return 0
    else:
        return label + 1 - dai_dict[plant_type]['dai_offset']


def get_data_by_split(data_dir, cv_split=0):
    """
    Loads the relevant train test split of the cv, split txt files.
    :return:
        filenames_allfiles: np array of filepaths of all files in train and test set
        y: np array of ints, 0 for healthy, 1 for sick
        train_ids: np array of ints, indicating which file name in filenames_allfiles belongs to a training sample
        test_ids: np_array of ints, indicating which file name in filenames_allfiles belongs to a test sample
        train_samples: list of strings, containing the sample id of all training samples
        test_samples: list of strings, containing the sample id of all test samples
    """
    # list of train and test sample ids for corresponding cv run
    with open(f"{data_dir}/rgb_dataset_splits/train_{cv_split}.txt") as f:
        train_samples = f.read().splitlines()
    with open(f"{data_dir}/rgb_dataset_splits/test_{cv_split}.txt") as f:
        test_samples = f.read().splitlines()

    # list of filenames
    filenames_allfiles = []
    train_ids = []
    test_ids = []
    fp_data_dir = data_dir + '/rgb_data'
    for _i, sample_id in enumerate(train_samples):
        filenames_allfiles.append(f"{fp_data_dir}/{sample_id[0]}/{sample_id}.JPEG")
        train_ids.append(_i)
    last_index_train = len(train_samples)
    for _i, sample_id in enumerate(test_samples):
        filenames_allfiles.append(f"{fp_data_dir}/{sample_id[0]}/{sample_id}.JPEG")
        test_ids.append(_i + last_index_train)

    assert len(train_samples) == len(train_ids)
    assert len(test_samples) == len(test_ids)

    filenames_allfiles = np.array(filenames_allfiles)
    train_ids = np.array(train_ids)
    test_ids = np.array(test_ids)

    # get label list
    y = []
    for fname in filenames_allfiles:
        fname = fname.split(".JPEG")[0].split("/")[-1].replace("_", ",")
        y.append(get_dai_label(fname))
    y = np.array(y)
    y[y > 0] = 1

    return filenames_allfiles, y, train_ids, test_ids, train_samples, test_samples


# Use .frombytes instead of .fromarray.
# This is >2x faster than img_grey
def img_frombytes(data):
    """
    See https://stackoverflow.com/questions/50134468/convert-boolean-numpy-array-to-pillow-image
    :param data:
    :return:
    """
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)


def reshape_flattened_to_tensor_rgb(data, width_height):
    """
    Reshape X (flattened data) to original size and swap axis to be conform with
    pytorch rgb tensors.
    """
    data = np.reshape(data, (data.shape[0], width_height, width_height, 3))
    data = np.swapaxes(data, 1, 3)
    data = np.swapaxes(data, 2, 3)
    return data


def resize(np_arr, img_size):
    img = Image.fromarray(np_arr)
    img = img.resize((img_size, img_size))
    return np.array(img)


def enhance(np_arr):
    img = Image.fromarray(np_arr)
    enhancer = ImageEnhance.Brightness(img)
    return np.array(enhancer.enhance(factor=2))


class PlantRGBDataset(DatasetWithSpecifications):
    """
    Make sure the following datasets are in place in the folder defined by data_dir in the constructor.
    1. preprocessed_masks.pyu pickle file in /mask subdir
    2. image files in /rgb_data
    3. cv train-test split files in /rgb_dataset_splits
    """
    def __init__(self, data_dir):
        """
        :param data_dir: The folder containing dataset
        :param split: train/dev/test string
        """
        rng = np.random.default_rng(seed=42)
        self.filenames, self.all_labels, self.train_ids, self.test_ids, self.train_samples, self.test_samples = \
            get_data_by_split(data_dir)

        # load the mask dictionary
        mask_data_dir = data_dir + '/mask/preprocessed_masks.pyu'
        with open(mask_data_dir, 'rb') as handle:
            self.mask_dict = pickle.load(handle)

        train_idxs, test_idxs = self.train_ids, self.test_ids
        train_idxs = rng.permutation(train_idxs)
        train_ln = int(len(train_idxs)*0.9)
        train_idxs, val_idxs = train_idxs[:train_ln], train_idxs[train_ln:]

        self.split_dict = {'train': train_idxs, 'val': val_idxs, 'test': test_idxs}
        all_train = np.stack([self[idx][0] for idx in train_idxs], axis=0)
        self.per_channel_avg = all_train.reshape([-1, 3]).mean(axis=0).astype(np.uint8)
        self.train_mean = all_train.reshape([-1, 3]).astype(np.float32).mean(axis=0)/255
        self.train_std = all_train.reshape([-1, 3]).astype(np.float32).std(axis=0)/255

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        """
        img : (rescale_size, rescale_size, 3)
        mask : (1, )
        """
        img = Image.open(self.filenames[idx])
        img = np.array(img)

        key = self.filenames[idx].split('/')[-1].split('.')[0]
        mask = self.mask_dict[key]
        mask = img_frombytes(mask)
        # invert mask as rrr loss need a mask where it is 0 in that region where the model should focus
        mask = np.array(np.logical_not(mask))

        val_test = (idx in self.split_dict['val']) or (idx in self.split_dict['test'])
        if val_test:
            _mask = np.expand_dims(mask, axis=-1).astype(np.uint8)
            img = img * (1 - _mask) + self.per_channel_avg * _mask

        # _y = self.all_labels[idx]
        # color = np.array([_y*255, (1-_y)*255, 0]).astype(np.uint8)
        # _mask = np.expand_dims(mask, axis=-1).astype(np.uint8)
        # img = img * (1 - _mask) + color * _mask

        img, mask = resize(img, 224), resize(mask, 224)
        img = enhance(img)
        group = self.all_labels[idx]

        # torch behaves weirdly when changing type from boolean. For instance, bool->long changes the scale of value
        # from 0-1->0-255
        mask = mask.astype(np.float32)
        mask /= max(mask.max(), 1e-5)

        return img, self.all_labels[idx], mask, group

    def targets(self):
        return np.array(self.all_labels)

    @property
    def num_classes(self) -> int:
        return 2
