import copy
import logging
import warnings
from typing import Any, Dict, List, Tuple

import torch
import torch.utils.data as data
import tenseal as ts
import numpy as np


from utils.util_crypto import context_ckks
from utils.util_model import (
    extract_parameters,
    flatten_model_parameters,
    load_model_from_parameters,
    get_gaussian_noise,
    get_laplace_noise,
)
from utils.util_sys import wrap_torch_median
from utils.util_sys import wrap_torch_sort

from utils.util_logger import logger

warnings.filterwarnings("ignore", category=UserWarning, module="tenseal")


def fusion_avg(model_updates: Dict[int, torch.nn.Module]) -> Dict[str, torch.Tensor]:
    avgerage_params = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            weighted_params = torch.zeros_like(
                next(iter(model_updates.values())).state_dict()[key].float()
            )
            for _, model in model_updates.items():
                param = model.state_dict()[key].float()
                weighted_params += param * 1.0 / len(model_updates)
            avgerage_params[key] = weighted_params

    return avgerage_params


def fusion_fedavg(
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
) -> Dict[str, torch.Tensor]:

    total_data_size = sum(data_size.values())
    weighted_avg_params = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            weighted_params = torch.zeros_like(
                next(iter(model_updates.values())).state_dict()[key].float()
            )
            for client_id, model in model_updates.items():
                weight = data_size[client_id] / total_data_size
                param = model.state_dict()[key].float()
                weighted_params += param * weight
            weighted_avg_params[key] = weighted_params

    return weighted_avg_params


def fusion_krum(
    model_updates: Dict[int, torch.nn.Module],
    max_expected_adversaries=1,
    device=torch.device("cpu"),
) -> Dict[str, torch.Tensor]:

    with torch.no_grad():
        ids = list(model_updates.keys())
        updates = [extract_parameters(model_updates[id]) for id in ids]
        updates = [update.to(device) for update in updates]
        num_updates = len(updates)
        updates_stack = torch.stack(updates)

        dist_matrix = torch.cdist(updates_stack, updates_stack, p=2)
        values, indices = torch.topk(
            dist_matrix,
            k=num_updates - max_expected_adversaries - 1,
            dim=1,
            largest=False,
            sorted=True,
        )
        # logger.debug(f"current krum values: {values}")
        scores = values.sum(dim=1)
        # logger.debug(f"current krum scores: {scores}")
        min_indices = torch.argmin(scores).item()
        logger.debug(f"current krum min index: {min_indices}")
        selected_id = ids[min_indices]
        logger.info(f"selected client id: {selected_id}")

    selected_model = model_updates[selected_id]
    krum_params = selected_model.state_dict()
    return krum_params


def fusion_median(
    model_updates: Dict[int, torch.nn.Module],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    median_params = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            params = torch.stack(
                [model.state_dict()[key].float() for model in model_updates.values()]
            )
            median_params[key] = wrap_torch_median(params, dim=0, device=device)

    return median_params


def fusion_clipping_median(
    model_updates: Dict[int, torch.nn.Module],
    clipping_threshold=0.1,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    median_params = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            params = torch.stack(
                [model.state_dict()[key].float() for model in model_updates.values()]
            )
            median_params[key] = wrap_torch_median(params, dim=0, device=device)
            median_params[key] = torch.clamp(
                median_params[key], -clipping_threshold, clipping_threshold
            )

    return median_params


def fusion_trimmed_mean(
    model_updates: Dict[int, torch.nn.Module],
    trimmed_ratio: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    trimmed_mean_params = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            params = torch.stack(
                [model.state_dict()[key].float() for model in model_updates.values()]
            )
            lower = int(params.size(0) * trimmed_ratio)
            upper = int(params.size(0) * (1 - trimmed_ratio))
            params = wrap_torch_sort(params, dim=0, device=device)[lower:upper]
            trimmed_mean_params[key] = torch.mean(params, dim=0)

    return trimmed_mean_params


def fusion_cos_defense(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    similarity_threshold: float = None,
) -> Dict[str, Any]:

    global_last_layer = list(global_model.parameters())[-2].view(-1)
    models = list(model_updates.values())
    last_layers = [list(model.parameters())[-2].view(-1) for model in models]

    with torch.no_grad():
        scores = torch.abs(
            torch.nn.functional.cosine_similarity(
                torch.stack(last_layers),
                global_last_layer,
            )
        )
        # print(scores)
        logger.info(f"current fusion scores: {scores}")
        min_score = torch.min(scores)
        scores = (scores - min_score) / (torch.max(scores) - min_score)
        logger.info(f"normalized fusion scores: {scores}")

        if similarity_threshold is None:
            similarity_threshold = torch.mean(scores)
        logger.info(f"similarity threshold: {similarity_threshold}")

        benign_indices = scores >= similarity_threshold
        if torch.sum(benign_indices) == 0:
            logger.warning("No models are considered benign based on the threshold.")
            logger.warning("Return global model of last round.")
            return global_model.state_dict()

        logger.info(f"current round client list: {model_updates.keys()}")
        logger.info(f"potential malicide indices: {benign_indices}")
        logger.info(f"checked benign indices: {benign_indices}")

        weight = 1 / torch.sum(benign_indices).float()
        fractions = benign_indices.float() * weight
        logger.info(f"current fusion fractions: {fractions}")

        weighted_params = copy.deepcopy(global_model.state_dict())
        for param_key in weighted_params.keys():
            temp_param = torch.zeros_like(
                global_model.state_dict()[param_key], dtype=torch.float32
            )
            for model, fraction in zip(models, fractions):
                temp_param += model.state_dict()[param_key] * fraction
            weighted_params[param_key].copy_(temp_param)
            # OUR OPTIMIZATION FOR DEFENSE
            # weighted_params[param_key] = torch.clamp(
            #     weighted_params[param_key], -0.1, 0.1
            # )

    return weighted_params


def fusion_dual_defense(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
    similarity_threshold: float = None,
    epsilon: float = None,
) -> Dict[str, torch.Tensor]:
    # simulate the hyper guard defense (privacy-preserving robust aggregation)
    # 1) each client generates pre-preprocessed update
    global_last_layer = list(global_model.parameters())[-2].view(-1)
    last_layers = {
        client_id: list(model.parameters())[-2].view(-1)
        for client_id, model in model_updates.items()
    }
    mormalized_global = global_last_layer / torch.norm(global_last_layer)
    normalized_locals = {
        client_id: last_layer / torch.norm(last_layer)
        for client_id, last_layer in last_layers.items()
    }
    # 2) encrypt and send to the fusion server
    encrypted_global = ts.ckks_vector(
        context_ckks, mormalized_global.flatten().tolist()
    )
    encrypted_locals = {
        client_id: ts.ckks_vector(context_ckks, normalized_local.flatten().tolist())
        for client_id, normalized_local in normalized_locals.items()
    }
    encrypted_updates = {}
    for client_id, model in model_updates.items():
        flattened_parameters = flatten_model_parameters(model)
        encrypted_parameters = [
            ts.ckks_vector(context_ckks, param) for param in flattened_parameters
        ]
        encrypted_updates[client_id] = encrypted_parameters

    # 3) server fuse encrypted update and return back the encrypted scores to each client

    # encrypted_global_dp = encrypted_global + get_gaussian_noise(
    #     encrypted_global.size(),
    #     epsilon=0.5,
    #     delta=1.0 / encrypted_global.size(),
    #     sensitivity=1,
    # )

    # encrypted_global_dp = encrypted_global + get_laplace_noise(
    #     encrypted_global.size(), epsilon=0.5, sensitivity=1
    # )

    if epsilon is not None and isinstance(epsilon, float):
        gaussian_nosie = get_gaussian_noise(
            1, epsilon=epsilon, delta=1.0 / encrypted_global.size(), sensitivity=1
        )
        encrypted_global = (
            encrypted_global + gaussian_nosie.tolist() * encrypted_global.size()
        )

    # simulate approximated clamp
    # encrypted_locals_clamp = {
    #     client_id: encrypted_local * 0.1
    #     for client_id, encrypted_local in encrypted_locals.items()
    # }

    encrypted_scores = {
        client_id: encrypted_local.dot(encrypted_global)
        for client_id, encrypted_local in encrypted_locals.items()
    }

    # 4) each client decrypt the scores and send back the benigns for validation
    client_selections = {}
    for client_id in model_updates.keys():
        scores = {
            client_id: np.abs(encrypted_score.decrypt())
            for client_id, encrypted_score in encrypted_scores.items()
        }
        logger.debug(f"client {client_id} scores: {scores}")
        min_score = np.min(list(scores.values()))
        max_score = np.max(list(scores.values()))
        diff_score = max_score - min_score
        scores_norm = {
            client_id: (score - min_score) / diff_score
            for client_id, score in scores.items()
        }
        logger.debug(f"client {client_id} norm scores: {scores_norm}")
        if similarity_threshold is None:
            similarity_threshold = np.mean(list(scores_norm.values()))
        logger.debug(f"client {client_id} similarity threshold: {similarity_threshold}")
        selected_benigns = [
            id for id, score in scores_norm.items() if score >= similarity_threshold
        ]
        logger.info(f"client {client_id} selected fusion benigns: {selected_benigns}")
        if len(selected_benigns) == 0:
            raise ValueError("No models are considered benign based on the threshold.")
        client_selections[client_id] = selected_benigns

    # 5) server counts and find the majority beningn selections
    count = {}
    for _, benigns in client_selections.items():
        _tuple = tuple(benigns)
        if _tuple in count:
            count[_tuple] += 1
        else:
            count[_tuple] = 1
    benigns = None
    max_count = 0
    for _benigns, _cnt in count.items():
        if _cnt > max_count:
            max_count = _cnt
            benigns = _benigns

    # 6) final secure aggregation
    logger.debug(f"final fusion benigns: {benigns}")
    total_size = sum(data_size[benign_id] for benign_id in benigns)
    fused_enc_params = [0] * len(encrypted_updates[benigns[0]])
    for benign_id in benigns:
        enc_param = encrypted_updates[benign_id]
        fusion_weight = data_size[benign_id] / total_size
        weighted_enc_param = [_p * fusion_weight for _p in enc_param]
        fused_enc_params = [x + y for x, y in zip(fused_enc_params, weighted_enc_param)]

    # 7) send to client for decryption
    _params = [param.decrypt() for param in fused_enc_params]
    fused_model = load_model_from_parameters(_params, global_model)
    fused_params = fused_model.state_dict()

    return fused_params
