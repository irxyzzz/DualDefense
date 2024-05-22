import copy
import random
from typing import Dict, Tuple

import torch
import pytest

from utils.util_fusion import fusion_avg
from utils.util_fusion import fusion_fedavg
from utils.util_fusion import fusion_cos_defense
from utils.util_fusion import fusion_hyper_guard
from utils.util_fusion import fusion_krum
from utils.util_fusion import fusion_median
from utils.util_fusion import fusion_clipping_median
from utils.util_fusion import fusion_trimmed_mean
from utils.models import TestNet


@pytest.fixture(scope="session")
def model_updates() -> Dict[int, torch.nn.Module]:
    size_parties = 5
    updates = {i + 1: TestNet() for i in range(size_parties)}
    for model in updates.values():
        for param in model.parameters():
            param.data.uniform_(0, 1)
    return updates


@pytest.fixture(scope="session")
def same_model() -> Tuple[torch.nn.Module, Dict[int, torch.nn.Module]]:
    model = TestNet()
    size_parties = 5
    updates = {i + 1: copy.deepcopy(model) for i in range(size_parties)}
    return model, updates


@pytest.fixture(scope="session")
def global_model() -> torch.nn.Module:
    model = TestNet()
    for param in model.parameters():
        param.data.uniform_(0, 1)
    return model


@pytest.fixture(scope="session")
def data_size(model_updates: Dict[int, torch.nn.Module]) -> Dict[int, int]:
    data_size = {id: random.randint(100, 200) for id in model_updates.keys()}
    return data_size


def test_fusion_avg(
    global_model: torch.nn.Module, model_updates: Dict[int, torch.nn.Module]
) -> None:
    result = fusion_avg(model_updates)

    assert isinstance(result, Dict)
    assert all(isinstance(key, str) for key in result.keys())
    assert all(isinstance(value, torch.Tensor) for value in result.values())

    average_model = copy.deepcopy(global_model)
    for param_key in global_model.state_dict().keys():
        temp_param = torch.zeros_like(global_model.state_dict()[param_key])
        for model in model_updates.values():
            temp_param += model.state_dict()[param_key]
        temp_param = temp_param / len(model_updates)
        average_model.state_dict()[param_key].copy_(temp_param)

    for _result, _expected_params in zip(
        result.values(), average_model.state_dict().values()
    ):
        assert torch.allclose(_result, _expected_params)


def test_fusion_fedavg(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
) -> None:
    result = fusion_fedavg(model_updates, data_size)

    assert isinstance(result, Dict)
    assert all(isinstance(key, str) for key in result.keys())
    assert all(isinstance(value, torch.Tensor) for value in result.values())

    fedavg_model = copy.deepcopy(global_model)
    for param_key in global_model.state_dict().keys():
        temp_param = torch.zeros_like(global_model.state_dict()[param_key])
        for pid, model in model_updates.items():
            temp_param += (
                data_size[pid]
                / sum(data_size.values())
                * model.state_dict()[param_key].float()
            )
        fedavg_model.state_dict()[param_key].copy_(temp_param)

    for _result, _expected_params in zip(
        result.values(), fedavg_model.state_dict().values()
    ):
        assert torch.allclose(_result, _expected_params)


def test_fusion_cos_defense(
    global_model: torch.nn.Module, model_updates: Dict[int, torch.nn.Module]
) -> None:
    result = fusion_cos_defense(global_model, model_updates)

    assert isinstance(result, Dict)
    assert all(isinstance(key, str) for key in result.keys())
    assert all(isinstance(value, torch.Tensor) for value in result.values())


def test_fusion_hyper_guard(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
) -> None:
    result = fusion_hyper_guard(global_model, model_updates, data_size)

    assert isinstance(result, Dict)
    assert all(isinstance(key, str) for key in result.keys())
    assert all(isinstance(value, torch.Tensor) for value in result.values())


def test_fusion_krum(model_updates: Dict[int, torch.nn.Module]) -> None:
    result = fusion_krum(model_updates)

    assert isinstance(result, Dict)
    assert all(isinstance(key, str) for key in result.keys())
    assert all(isinstance(value, torch.Tensor) for value in result.values())


def test_fusion_median(model_updates: Dict[int, torch.nn.Module]) -> None:
    result = fusion_median(model_updates)

    assert isinstance(result, Dict)
    assert all(isinstance(key, str) for key in result.keys())
    assert all(isinstance(value, torch.Tensor) for value in result.values())


def test_fusion_clipping_median(model_updates: Dict[int, torch.nn.Module]) -> None:
    result = fusion_clipping_median(model_updates)

    assert isinstance(result, Dict)
    assert all(isinstance(key, str) for key in result.keys())
    assert all(isinstance(value, torch.Tensor) for value in result.values())


def test_fusion_trimmed_mean(model_updates: Dict[int, torch.nn.Module]) -> None:
    result = fusion_trimmed_mean(model_updates)

    assert isinstance(result, Dict)
    assert all(isinstance(key, str) for key in result.keys())
    assert all(isinstance(value, torch.Tensor) for value in result.values())
