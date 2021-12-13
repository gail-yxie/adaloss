import numpy as np
from PIL import Image
import torchvision


def get_cifar100(root, cfg_trainer, train=True,
                 transform_train=None, transform_val=None,
                 download=True):
    base_dataset = torchvision.datasets.CIFAR100(root, train=train, download=download)
    if train:
        train_idxs, val_idxs = train_val_split(base_dataset.targets)
        train_dataset = CIFAR100_train(root, cfg_trainer, train_idxs, train=True, transform=transform_train)
        val_dataset = CIFAR100_val(root, cfg_trainer, val_idxs, train=train, transform=transform_val)

        print(f"Train: {len(train_dataset)} Val: {len(val_dataset)}")  # Train: 45000 Val: 5000
    else:
        train_dataset = []
        val_dataset = CIFAR100_val(root, cfg_trainer, None, train=train, transform=transform_val)
        print(f"Test: {len(val_dataset)}")

    return train_dataset, val_dataset


def train_val_split(base_dataset: torchvision.datasets.CIFAR100):
    num_classes = 100
    base_dataset = np.array(base_dataset)
    train_n = int(len(base_dataset) * 0.9 / num_classes)
    train_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(base_dataset == i)[0]
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:train_n])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs


class CIFAR100_train(torchvision.datasets.CIFAR100):
    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_train, self).__init__(root, train=train,
                                             transform=transform, target_transform=target_transform,
                                             download=download)
        self.num_classes = 100
        self.cfg_trainer = cfg_trainer
        self.train_data = self.data[indexs]
        self.train_labels = np.array(self.targets)[indexs]
        self.indexs = indexs

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.train_data)


class CIFAR100_val(torchvision.datasets.CIFAR100):

    def __init__(self, root, cfg_trainer, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_val, self).__init__(root, train=train,
                                           transform=transform, target_transform=target_transform,
                                           download=download)

        self.num_classes = 100
        self.cfg_trainer = cfg_trainer
        if train:
            self.train_data = self.data[indexs]
            self.train_labels = np.array(self.targets)[indexs]
        else:
            self.train_data = self.data
            self.train_labels = np.array(self.targets)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
