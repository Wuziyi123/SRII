import torch
from PIL import Image
import pandas as pd
import os
import os.path
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple
import json
import matplotlib.pyplot as plt
from pandas import DataFrame

from torch.utils.data import Dataset


def read_csv_classes(csv_dir: str, csv_name: str):
    data = pd.read_csv(os.path.join(csv_dir, csv_name))
    # print(data.head(1))  # filename, label

    label_set = set(data["label"].drop_duplicates().values)

    # print("{} have {} images and {} classes.".format(csv_name,
    #                                                  data.shape[0],
    #                                                  len(label_set)))
    return data, label_set


def calculate_split_info(path: str, label_dict: dict, root: str, train: bool = True, rate: float = 0.2):
    targets = []
    # read all images
    image_dir = os.path.join(path, "images")
    images_list = [i for i in os.listdir(image_dir) if i.endswith(".jpg")]
    # print("find {} images in dataset.".format(len(images_list)))

    # train_data = (38400, 2)  train_label = 64
    train_data, train_label = read_csv_classes(path, "train.csv")
    val_data, val_label = read_csv_classes(path, "val.csv")
    test_data, test_label = read_csv_classes(path, "test.csv")

    # Union operation
    labels = (train_label | val_label | test_label)
    labels = list(labels)
    origin_labels = labels.copy()
    # print("all classes: {}".format(len(origin_labels)))
    class_to_idx = {_class: i for i, _class in enumerate(origin_labels)}

    # labels.sort()
    # print("all classes: {}".format(len(labels)))
    # # create classes_name.json
    # classes_label = dict([(label, [index, label_dict[label]]) for index, label in enumerate(labels)])
    # json_str = json.dumps(classes_label, indent=4)
    # with open('classes_name.json', 'w') as json_file:
    #     json_file.write(json_str)

    # concat csv data = (60000, 2)
    data = pd.concat([train_data, val_data, test_data], axis=0)
    # print("total data shape: {}".format(data.shape))

    # shuffle_data = data.sample(frac=1, random_state=1)
    # target = shuffle_data["label"].values
    # targets = [class_to_idx[i] for i in target]
    # shuffle_data = shuffle_data["filename"].values

    # split data on every classes
    num_every_classes = []
    split_train_data = []
    split_val_data = []
    for label in labels:
        class_data = data[data["label"] == label]
        num_every_classes.append(class_data.shape[0])

        # shuffle
        shuffle_data = class_data.sample(frac=1, random_state=1)
        num_train_sample = int(class_data.shape[0] * (1 - rate))
        split_train_data.append(shuffle_data[:num_train_sample])
        split_val_data.append(shuffle_data[num_train_sample:])

    train_data = split_train_data[0]
    test_data = split_val_data[0]
    for i in range(1, len(split_train_data)):
        train_data = pd.concat([train_data, split_train_data[i]], axis=0)
        test_data = pd.concat([test_data, split_val_data[i]], axis=0)

    if train:
        # shuffle_data = data.sample(frac=1, random_state=1)
        target = train_data["label"].values
        targets = [class_to_idx[i] for i in target]
        shuffle_data = train_data["filename"].values
    else:
        # shuffle_data = data.sample(frac=1, random_state=1)
        target = test_data["label"].values
        targets = [class_to_idx[i] for i in target]
        shuffle_data = test_data["filename"].values

    shuffle_data = np.array([os.path.join(root, i) for i in shuffle_data], dtype=str)

    return shuffle_data, targets


class ImageNet100(Dataset):

    def __init__(
            self,
            root: str,
            train: bool = True
    ) -> None:

        super(ImageNet100, self).__init__()

        self.train = train  # training set or test set

        self.data: Any = []
        self.targets = []

        data_dir = "mini-imagenet/"  # 指向数据集的根目录
        json_path = "mini-imagenet/imagenet_class_index.json"  # 指向imagenet的索引标签文件

        # load imagenet labels
        label_dict = json.load(open(json_path, "r"))
        label_dict = dict([(v[0], v[1]) for k, v in label_dict.items()])

        self.data, self.targets = calculate_split_info(data_dir, label_dict, root, self.train)

        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        # self._load_meta()


    # def _load_meta(self) -> None:
    #     path = os.path.join(self.root, self.base_folder, self.meta['filename'])
    #     with open(path, 'rb') as infile:
    #         data = pickle.load(infile, encoding='latin1')
    #         self.classes = data[self.meta['key']]
    #     self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


