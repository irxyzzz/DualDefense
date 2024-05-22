from typing import List
import math
import copy

import torch
import numpy as np

from utils.models import ResNet18, MNISTCNN, FashionMNISTCNN


def get_client_model(dataset: str, num_parties: int, device: torch.device) -> dict:
    """
    Returns the client models based on the dataset.

    Args:
        dataset (str): The dataset used for training.
        num_parties (int): The number of parties in the federated learning system.
        device (torch.device): The device to use for the model.
    Returns:
        dict: A dictionary containing the client models.
    """

    client_models = {id: None for id in range(num_parties)}
    for client_id in range(num_parties):
        if dataset == "mnist":
            model = MNISTCNN()
        elif dataset == "fmnist":
            model = FashionMNISTCNN()
        elif dataset == "cifar10" or dataset == "svhn":
            model = ResNet18()
        else:
            raise ValueError("Invalid dataset")

        if model is not None:
            model.to(device)
        client_models[client_id] = model

    return client_models


def get_server_model(dataset: str, device: torch.device) -> torch.nn.Module:
    """
    Returns the server model based on the dataset.

    Args:
        dataset (str): The dataset used for training.
        device (torch.device): The device to use for the model.
    Returns:
        torch.nn.Module: The server model.
    """
    if dataset == "mnist":
        model = MNISTCNN()
    elif dataset == "fmnist":
        model = FashionMNISTCNN()
    elif dataset == "cifar10" or dataset == "svhn":
        model = ResNet18()
    if model is not None:
        model.to(device)

    return model


def extract_parameters(model: torch.nn.Module) -> torch.Tensor:
    params = [p.view(-1) for p in model.parameters()]
    return torch.cat(params)


def flatten_model_parameters(model: torch.nn.Module) -> List[List[float]]:
    """
    Converts each layer's parameters of a PyTorch model into a one-dimensional array format and stores them in a list.

    Args:
        model (torch.nn.Module): The PyTorch model to process.

    Returns:
        List[List[float]]: A list containing one-dimensional arrays of parameters for each layer of the model.
    """
    flattened_parameters = [
        param.data.flatten().tolist() for param in model.parameters()
    ]

    return flattened_parameters


def load_model_from_parameters(
    flatten_parameters: List[List[float]], model: torch.nn.Module
) -> torch.nn.Module:
    """
    Recovers the model from the flattened parameters.

    Args:
        flatten_parameters (List[List[float]]): The flattened parameters of the model.
        model (torch.nn.Module): The model to recover.

    Returns:
        torch.nn.Module: The recovered model.
    """
    for param, flatten_param in zip(model.parameters(), flatten_parameters):
        param.data = torch.tensor(flatten_param).view(param.data.shape)

    return model


def get_gaussian_noise(
    size: int, epsilon: float = 0.5, delta: float = 1e-5, sensitivity: float = 1
) -> np.ndarray:

    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, size)

    return noise


def get_laplace_noise(
    size: int, epsilon: float = 0.5, sensitivity: float = 1
) -> np.ndarray:

    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size)
    return noise


def ipm_attack_craft_model(
    old_model, new_model, action: int = 5, b: int = -1
) -> torch.nn.Module:
    crafted_model = copy.deepcopy(old_model)

    for old_param, new_param, crafted_param in zip(
        old_model.parameters(), new_model.parameters(), crafted_model.parameters()
    ):
        weight_diff = old_param.data - new_param.data
        crafted_weight_diff = b * weight_diff * action
        crafted_param.data = old_param.data - crafted_weight_diff

    return crafted_model


def _fang_attack_compute_lambda(
    param_updates: torch.Tensor, param_global: torch.Tensor, n_attackers: int
) -> float:

    distances = []
    n_benign, d = param_updates.shape
    for update in param_updates:
        distance = torch.norm((param_updates - update), dim=1)
        distances = (
            distance[None, :]
            if not len(distances)
            else torch.cat((distances, distance[None, :]), 0)
        )

    distances[distances == 0] = 10000
    distances = torch.sort(distances, dim=1)[0]
    scores = torch.sum(distances[:, : n_benign - 2 - n_attackers], dim=1)
    min_score = torch.min(scores)
    term_1 = min_score / (
        (n_benign - n_attackers - 1) * torch.sqrt(torch.Tensor([d]))[0]
    )
    max_wre_dist = torch.max(torch.norm((param_updates - param_global), dim=1)) / (
        torch.sqrt(torch.Tensor([d]))[0]
    )

    return term_1 + max_wre_dist


def _fang_attack_multi_krum(
    param_updates: torch.Tensor, n_attackers: int, multi_k=False
):
    nusers = param_updates.shape[0]
    candidates = []
    candidate_indices = []
    remaining_updates = param_updates
    all_indices = np.arange(len(param_updates))

    while len(remaining_updates) > 2 * n_attackers + 2:
        distances = []
        for update in remaining_updates:
            distance = torch.norm((remaining_updates - update), dim=1) ** 2
            distances = (
                distance[None, :]
                if not len(distances)
                else torch.cat((distances, distance[None, :]), 0)
            )

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(
            distances[:, : len(remaining_updates) - 2 - n_attackers], dim=1
        )
        indices = torch.argsort(scores)[: len(remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = (
            remaining_updates[indices[0]][None, :]
            if not len(candidates)
            else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        )
        remaining_updates = torch.cat(
            (remaining_updates[: indices[0]], remaining_updates[indices[0] + 1 :]), 0
        )
        if not multi_k:
            break
    # print(len(remaining_updates))
    aggregate = torch.mean(candidates, dim=0)
    return aggregate, np.array(candidate_indices)


def fang_attack(
    param_updates: torch.Tensor,
    param_global: torch.Tensor,
    deviation: torch.Tensor,
    n_attackers: int,
):

    lamda = _fang_attack_compute_lambda(param_updates, param_global, n_attackers)

    threshold = 1e-5
    mal_update = []

    while lamda > threshold:
        mal_update = -lamda * deviation
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, param_updates), 0)

        # print(mal_updates.shape, n_attackers)
        agg_grads, krum_candidate = _fang_attack_multi_krum(
            mal_updates, n_attackers, multi_k=False
        )
        if krum_candidate < n_attackers:
            # print('successful lamda is ', lamda)
            return mal_update
        else:
            mal_update = []

        lamda *= 0.5

    if not len(mal_update):
        mal_update = param_global - lamda * deviation

    return mal_update


def aggregation_tailored_attack(
    param_updates: torch.Tensor,
    param_global: torch.Tensor,
    n_attackers,
    dev_type="sign",
):

    if dev_type == "unit_vec":
        deviation = param_global / torch.norm(
            param_global
        )  # unit vector, dir opp to good dir
    elif dev_type == "sign":
        deviation = torch.sign(param_global)
    elif dev_type == "std":
        deviation = torch.std(param_updates, 0)

    lamda = torch.Tensor(
        [20.0]
    ).cuda()  # compute_lambda_our(all_updates, model_re, n_attackers)

    threshold_diff = 1e-5
    lamda_fail = lamda
    lamda_succ = 0

    while torch.abs(lamda_succ - lamda) > threshold_diff:
        mal_update = param_global - lamda * deviation
        mal_updates = torch.stack([mal_update] * n_attackers)
        mal_updates = torch.cat((mal_updates, param_updates), 0)

        agg_grads, krum_candidate = _fang_attack_multi_krum(
            mal_updates, n_attackers, multi_k=True
        )

        if np.sum(krum_candidate < n_attackers) == n_attackers:
            # print('successful lamda is ', lamda)
            lamda_succ = lamda
            lamda = lamda + lamda_fail / 2
        else:
            lamda = lamda - lamda_fail / 2

        lamda_fail = lamda_fail / 2

    mal_update = param_global - lamda_succ * deviation

    return mal_update


def scaling_attack(model_update: torch.nn.Module):
    params_list = list(model_update.parameters())
    num_params = len(params_list)

    for param in params_list:
        param.data *= num_params
    return model_update


def alie_attack(model_update: torch.nn.Module, epsilon=0.1):
    with torch.no_grad():
        for param in model_update.parameters():
            param.add_(torch.randn(param.size(), device=param.device) * epsilon)
    return model_update
