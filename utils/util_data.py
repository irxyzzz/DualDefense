import os
import copy
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import numpy as np
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

from utils.datasets import CustomSVHN

from utils.util_sys import create_folder_if_not_exists

from utils.util_logger import logger


def load_data_cifar10(
    dir_data: str,
) -> Tuple[CIFAR10, CIFAR10]:

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    cifar10_train_ds = CIFAR10(dir_data, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10(dir_data, train=False, download=True, transform=transform)

    return cifar10_train_ds, cifar10_test_ds


def load_data_mnist(
    dir_data: str,
) -> Tuple[MNIST, MNIST]:

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    mnist_train_ds = MNIST(dir_data, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST(dir_data, train=False, download=True, transform=transform)

    return mnist_train_ds, mnist_test_ds


def load_data_fmnist(
    dir_data: str,
) -> Tuple[FashionMNIST, FashionMNIST]:

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    fmnist_train_ds = FashionMNIST(
        dir_data, train=True, download=True, transform=transform
    )
    fmnist_test_ds = FashionMNIST(
        dir_data, train=False, download=True, transform=transform
    )

    return fmnist_train_ds, fmnist_test_ds


def load_data_svhn(
    dir_data: str,
) -> Tuple[CustomSVHN, CustomSVHN]:

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    svhn_train_ds = CustomSVHN(
        dir_data, split="train", download=True, transform=transform
    )
    svhn_test_ds = CustomSVHN(
        dir_data, split="test", download=True, transform=transform
    )

    return svhn_train_ds, svhn_test_ds


def get_client_data(
    dataset: str,
    dir_data: str,
    num_clients: int,
    partition_type: str,
    partition_beta: float = 0.5,
) -> Dict[int, Tuple[data.Dataset, data.Dataset]]:

    if dataset == "mnist":
        train_dataset, test_dataset = load_data_mnist(dir_data)
    elif dataset == "fmnist":
        train_dataset, test_dataset = load_data_fmnist(dir_data)
    elif dataset == "cifar10":
        train_dataset, test_dataset = load_data_cifar10(dir_data)
    elif dataset == "svhn":
        train_dataset, test_dataset = load_data_svhn(dir_data)
    else:
        raise ValueError("Dataset not supported")

    y_train = train_dataset.targets
    if y_train is not None:
        num_samples = len(y_train)
        num_classes = len(np.unique(y_train))
    else:
        raise ValueError("Cannot acqurie the number of samples and classes")

    if partition_type == "iid":
        logger.info("partitioning the data into IID setting")
        num_items_per_client = num_samples // num_clients
        remain = num_samples % num_clients
        lengths = [num_items_per_client] * num_clients
        lengths[-1] += remain
        client_datasets = data.random_split(train_dataset, lengths)
        client_datasets = {
            i: (client_datasets[i], test_dataset) for i in range(num_clients)
        }

    elif partition_type == "noniid":
        logger.info("partitioning the data into non-IID setting")

        class_indices = {
            i: np.where(np.array(train_dataset.targets) == i)[0]
            for i in range(num_classes)
        }
        client_indices = {cid: [] for cid in range(num_clients)}

        for c in range(num_classes):
            proportions = np.random.dirichlet(np.repeat(partition_beta, num_clients))
            proportions = proportions / proportions.sum() * len(class_indices[c])
            proportions = np.cumsum(proportions).astype(int)
            splits = np.split(class_indices[c], proportions[:-1])
            for cid in range(num_clients):
                client_indices[cid].extend(splits[cid].tolist())

        client_datasets = {
            cid: (get_full_subset(train_dataset, indices), test_dataset)
            for cid, indices in client_indices.items()
        }
    else:
        raise ValueError("Partition type not supported")

    record_data_statistic(client_datasets)

    return client_datasets


def get_global_test_data_loader(
    dataset: str,
    dir_data: str,
    batch_size: int,
) -> data.DataLoader:
    if dataset == "mnist":
        _dir = os.path.join(dir_data, "mnist/")
        _, test_dataset = load_data_mnist(_dir)
    elif dataset == "fmnist":
        _dir = os.path.join(dir_data, "fmnist/")
        _, test_dataset = load_data_fmnist(_dir)
    elif dataset == "cifar10":
        _dir = os.path.join(dir_data, "cifar10/")
        _, test_dataset = load_data_cifar10(_dir)
    elif dataset == "svhn":
        _dir = os.path.join(dir_data, "svhn/")
        _, test_dataset = load_data_svhn(_dir)
    else:
        raise ValueError("Dataset not supported")

    test_dl = create_data_loader(test_dataset, batch_size)
    return test_dl


def get_full_subset(dataset, indices):
    class SubDataset(data.Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
            self.targets = [dataset.targets[i] for i in indices]
            self.classes = getattr(dataset, "classes", None)

        def __getitem__(self, idx):
            # data, _ = self.dataset[self.indices[idx]]
            # return data, self.targets[idx]
            return self.dataset[self.indices[idx]]

        def __len__(self):
            return len(self.indices)

    return SubDataset(dataset, indices)


def get_client_data_loader(
    dataset: str,
    dir_data: str,
    num_clients: int,
    partition_type: str,
    partition_beta: float,
    batch_size: int,
) -> Dict[int, Tuple[data.DataLoader, data.DataLoader]]:

    client_datasets = None

    if dataset == "mnist":
        _dir = os.path.join(dir_data, "mnist/")

    elif dataset == "fmnist":
        _dir = os.path.join(dir_data, "fmnist/")
    elif dataset == "cifar10":
        _dir = os.path.join(dir_data, "cifar10/")
    elif dataset == "svhn":
        _dir = os.path.join(dir_data, "svhn/")
    else:
        raise ValueError("Dataset not supported")

    create_folder_if_not_exists(_dir)
    client_datasets = get_client_data(
        dataset, _dir, num_clients, partition_type, partition_beta
    )
    if client_datasets is None:
        raise ValueError("Partition dataset is not available")

    client_data_loader = {}
    for _cid, _dataset in client_datasets.items():
        train_ds, test_ds = _dataset[0], _dataset[1]
        train_dl = create_data_loader(train_ds, batch_size)
        test_dl = create_data_loader(test_ds, batch_size)

        client_data_loader[_cid] = (train_dl, test_dl)

    return client_data_loader


def create_data_loader(dataset: data.Dataset, batch_size: int, shuffle: bool = True):
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=shuffle,
    )


def record_data_statistic(client_datasets: dict) -> Dict[int, dict]:

    detailed_stats = {}
    for client_id, dataset in client_datasets.items():
        label_count = defaultdict(int)
        features = []
        for feature, label in dataset[0]:
            label_count[label] += 1
            features.append(feature)

        detailed_stats[client_id] = {
            "train_label_distribution": dict(label_count),
            "train_total_samples": len(dataset[0]),
        }
        logger.info(f"Client {client_id} - total training samples: {len(dataset[0])}")
        logger.info(f"Client {client_id} - training label distribution: {label_count}")

    return detailed_stats
