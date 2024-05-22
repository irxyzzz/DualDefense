import os
import warnings
import datetime
import logging

from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore", category=UserWarning, module="tenseal")


def setup_logger() -> logging.Logger:
    log_filename = os.getenv("LOG_FILE_NAME", None)
    if log_filename is None:
        raise ValueError("LOG_FILE_NAME is not set")

    logger = logging.getLogger("HyperGuard")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        # "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s"
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def setup_tensorboard(log_dir: str, log_file: str) -> SummaryWriter:
    tensorboard = SummaryWriter(os.path.join(log_dir, log_file))

    return tensorboard


logger = setup_logger()
