import pytest

import numpy as np
import torch
import torch.utils

from utils.util_data import load_data_cifar10
from utils.util_data import get_party_data
from utils.util_data import get_party_data_loader


def test_load_data_cifar10():
    dataset_train, dataset_test = load_data_cifar10("./data/cifar10/")
    assert len(dataset_train) == 50000
    assert len(dataset_test) == 10000
    assert len(dataset_train[0]) == 2
    assert isinstance(dataset_train.data, np.ndarray)
    assert isinstance(dataset_train.targets, list)


def test_get_party_data_iid():
    dataset = "cifar10"
    dir_data = "./data/cifar10/"
    num_clients = 5
    partition_type = "iid"

    client_datasets = get_party_data(dataset, dir_data, num_clients, partition_type)
    assert isinstance(client_datasets, dict)
    assert len(client_datasets) == num_clients

    for key, value in client_datasets.items():
        assert isinstance(key, int)
        assert isinstance(value, tuple)
        assert len(value) == 2


def test_get_party_data_noniid():
    dataset = "cifar10"
    dir_data = "./data/cifar10/"
    num_clients = 5
    partition_type = "noniid"
    partition_beta = 0.5

    client_datasets = get_party_data(
        dataset, dir_data, num_clients, partition_type, partition_beta
    )
    assert isinstance(client_datasets, dict)
    assert len(client_datasets) == num_clients

    for key, value in client_datasets.items():
        assert isinstance(key, int)
        assert isinstance(value, tuple)
        assert len(value) == 2
        assert isinstance(value[0], torch.utils.data.Dataset)


def test_get_party_data_loader():
    dataset = "cifar10"
    dir_data = "./data/"
    num_clients = 5
    partition_type = "noniid"
    partition_beta = 0.5

    batch_size = 128
    attacker_strategy_list = [
        "none",
        "untarget_label_flipping",
        "target_label_flipping",
    ]
    poison_ratio = 0.1
    attack_list = [0, 1]
    target_label = 7
    poison_label = 0

    for attacker_strategy in attacker_strategy_list:
        client_data_loader = get_party_data_loader(
            dataset,
            dir_data,
            num_clients,
            partition_type,
            partition_beta,
            batch_size,
            attacker_strategy,
            poison_ratio,
            attack_list,
            target_label,
            poison_label,
        )
        assert isinstance(client_data_loader, dict)
        assert len(client_data_loader) == num_clients
        for k, v in client_data_loader.items():
            assert isinstance(k, int)
            assert isinstance(v, tuple)
            assert isinstance(v[0], torch.utils.data.DataLoader)
