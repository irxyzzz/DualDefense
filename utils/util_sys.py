import os

import torch

from utils.util_logger import logger


def get_available_device(decice: str = None) -> torch.device:
    if decice is not None:
        logger.info("using the preferred device: {} for pythorch".format(decice))
        return torch.device(decice)
    else:
        logger.info(
            "no preferred device is specified, using the available device for pythorch"
        )
        _device = None
        if torch.backends.mps.is_available():
            _device = torch.device("mps")
        elif torch.cuda.is_available():
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")

        logger.info("using {} device for pythorch".format(_device.type))
        return _device


def create_folder_if_not_exists(folder_path: str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logger.info(f"Folder '{folder_path}' created.")
    else:
        logger.info(f"Folder '{folder_path}' already exists.")


def wrap_torch_median(
    tensor: torch.Tensor, dim: int = 0, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    if device == torch.device("mps"):
        return torch.median(tensor.to("cpu"), dim=dim).values.to(device)
    else:
        return torch.median(tensor, dim=dim).values


def wrap_torch_sort(
    tensor: torch.Tensor, dim: int = 0, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    if device == torch.device("mps"):
        return torch.sort(tensor.to("cpu"), dim=dim).values.to(device)
    else:
        return torch.sort(tensor, dim=dim).values


def intersection_of_lists(list1: list, list2: list) -> list:
    set1 = set(list1)
    set2 = set(list2)
    union_set = set1.intersection(set2)

    return list(union_set)
