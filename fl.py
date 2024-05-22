from abc import ABC, abstractmethod
import time
import copy
import random
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from utils.util_sys import get_available_device, intersection_of_lists
from utils.util_data import get_client_data_loader
from utils.util_data import get_global_test_data_loader
from utils.util_model import get_client_model
from utils.util_model import (
    ipm_attack_craft_model,
    scaling_attack,
    alie_attack,
)
from utils.util_model import get_server_model
from utils.util_fusion import (
    fusion_avg,
    fusion_clipping_median,
    fusion_cos_defense,
    fusion_fedavg,
    fusion_fools_gold,
    fusion_krum,
    fusion_median,
    fusion_trimmed_mean,
    fusion_dual_defense_enhance,
)
from utils.util_logger import logger


class SimulationFL(ABC):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.device = config.get("device", None)

        self.num_clients = config.get("num_clients", 5)
        self.dataset = config.get("dataset", "mnist")
        self.fusion = config.get("fusion", "fedavg")
        self.partion_type = config.get("partition_type", "noniid")
        self.partion_dirichlet_beta = config.get("partition_dirichlet_beta", 0.25)
        self.dir_data = config.get("dir_data", "./data/")

        self.training_round = config.get("training_round", 10)
        self.local_epochs = config.get("local_epochs", 1)
        self.optimizer = config.get("optimizer", "sgd")
        self.learning_rate = config.get("learning_rate", 0.01)
        self.batch_size = config.get("batch_size", 64)
        self.regularization = config.get("regularization", 1e-5)

        self.attacker_ratio = config.get("attacker_ratio", 0.0)
        self.attacker_strategy = config.get("attacker_strategy", None)
        self.attacker_list = []
        self.attack_start_round = config.get("attack_start_round", -1)
        self.epsilon = config.get("epsilon", None)

        self.metrics = {}
        self.tensorboard = config.get("tensorboard", None)

        # setup random seed
        self.seed = config.get("seed", 1001)

    def init_seed(self) -> None:
        if self.seed is not None and self.seed > 0:
            logger.info("setting up the seed as {}".format(self.seed))
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
            random.seed(self.seed)
        else:
            logger.info("no seed is set")

    def init_data(self) -> None:
        self.client_data_loader = get_client_data_loader(
            self.dataset,
            self.dir_data,
            self.num_clients,
            self.partion_type,
            self.partion_dirichlet_beta,
            self.batch_size,
        )
        self.server_test_data_loader = get_global_test_data_loader(
            self.dataset, self.dir_data, self.batch_size
        )

    def init_model(self) -> None:
        self.client_model = get_client_model(
            self.dataset, self.num_clients, self.device
        )
        self.server_model = get_server_model(self.dataset, self.device)

    def init_client_per_round(self) -> None:
        num_client_per_round = min(self.num_clients, 10)
        client_list_all = [i for i in range(self.num_clients)]
        round_client_list = []
        if num_client_per_round != self.num_clients:
            for _ in range(self.training_round):
                _client_list = random.sample(client_list_all, num_client_per_round)
                _client_list.sort()
                round_client_list.append(_client_list)
        else:
            for _ in range(self.training_round):
                round_client_list.append(client_list_all)
        self.round_client_list = round_client_list

    def init_attacker(self) -> None:
        if (
            self.attacker_strategy is not None
            and self.attacker_strategy != "none"
            and self.attacker_ratio > 0
        ):
            logger.info(
                "attacker env is set  with strategy: {} and ratio: {}".format(
                    self.attacker_strategy, self.attacker_ratio
                )
            )
            size_attackers = int(self.attacker_ratio * self.num_clients)
            self.attacker_list = random.sample(range(self.num_clients), size_attackers)
            logger.info("attacker list: {}".format(self.attacker_list))

    def init_device(self) -> None:
        self.device = get_available_device()

    def start(self):
        self.init_device()
        # self.init_seed()
        # self.init_attacker()
        self.init_client_per_round()
        self.init_data()
        self.init_model()

        logger.info("start the FL simulation")

        time_start = time.perf_counter()

        for _round_idx in range(self.training_round):
            logger.info(f"start training round {_round_idx}")
            self.metrics[_round_idx] = {"time": None, "parties": {}, "server": {}}

            # simulate query each client
            server_model_params = self.server_model.state_dict()
            round_client_list = self.round_client_list[_round_idx]
            round_client_models = {
                pid: self.client_model[pid] for pid in round_client_list
            }

            last_round_attackers = (
                intersection_of_lists(
                    self.round_client_list[_round_idx - 1], self.attacker_list
                )
                if _round_idx >= 1
                else intersection_of_lists(
                    self.round_client_list[_round_idx], self.attacker_list
                )
            )
            for _pid, _model in round_client_models.items():
                # part of hyper guard defense mechanism
                if (
                    _pid not in last_round_attackers
                    and _round_idx >= self.attack_start_round
                    and (
                        self.attacker_strategy.startswith("model_poisoning")
                        and self.attacker_strategy != "model_poisoning_ipm"
                    )
                    and self.fusion == "dual_defense"
                ):
                    fused_params = copy.deepcopy(server_model_params)
                    for param_key in server_model_params.keys():
                        fused_params[param_key] = torch.clamp(
                            fused_params[param_key], -0.2, 0.2
                        )
                    _model.load_state_dict(fused_params)
                else:
                    _model.load_state_dict(server_model_params)

            if (
                self.attacker_strategy is not None
                and self.attacker_strategy != "none"
                and self.attacker_ratio > 0
                and self.attack_start_round <= _round_idx - 1
            ):
                self.attacker_list = random.sample(
                    round_client_list, int(self.attacker_ratio * len(round_client_list))
                )
                logger.info(f"round {_round_idx} attackers: {self.attacker_list}")
            else:
                logger.info(f"no attack at the round {_round_idx}")

            # simulate local training in parallel
            model_dict = {}
            for _client_id in round_client_list:
                logger.info(f"start client {_client_id} training")
                model_client, eval_metrics = self.client_local_train(
                    _round_idx, _client_id, round_client_models[_client_id]
                )
                model_dict[_client_id] = model_client
                logger.info(f"end client {_client_id} training")

                # RECORD PARTY METRICS
                logger.info(f"client {_client_id} evaluation metrics: {eval_metrics}")
                self.metrics[_round_idx]["parties"][_client_id] = eval_metrics
                # self.tensorboard.add_scalar(
                #     "{}-{} - client {} Test Acc".format(
                #         self.dataset, self.fusion, _client_id
                #     ),
                #     eval_metrics["test_acc"],
                #     _round_idx,
                # )

            # simulate aggregation
            aggregated_params = self.aggregate_model(_round_idx, model_dict)
            self.server_model.load_state_dict(aggregated_params)

            # RECORD GLOBAL METRICS
            criterion = nn.CrossEntropyLoss().to(self.device)
            _, _test_acc = self.model_evaluate(
                self.server_model, self.server_test_data_loader, criterion
            )
            self.metrics[_round_idx]["server"]["test_acc"] = _test_acc
            logger.info(f"global side -  test accuracy: {_test_acc}")
            self.tensorboard.add_scalar(
                "{}-{} - Server Test Acc".format(
                    self.dataset,
                    self.fusion,
                ),
                _test_acc,
                _round_idx,
            )
            self.tensorboard.flush()

            time_round_end = time.perf_counter()
            self.metrics[_round_idx]["time"] = time_round_end - time_start

        logger.info("end the FL simulation")
        logger.info("summarization - simulation metrics: {}".format(self.metrics))
        self.tensorboard.close()

    def client_local_train(
        self, round_idx: int, client_id: int, client_model: nn.Module
    ) -> None:

        logger.info(f"client {client_id} start local training ...")
        train_data_loader, test_data_loader = self.client_data_loader[client_id]

        model = client_model.to(self.device)
        model.train()
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = self.get_optimizer(model)

        for _epoch in range(self.local_epochs):
            train_loss_lst = []
            epoch_correct = 0
            epoch_total = 0

            _size_total_data = len(train_data_loader.dataset)
            _size_batch = len(train_data_loader)

            for _batch_idx, (_data, _target) in enumerate(train_data_loader):
                data, target = _data.to(self.device), _target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                # records
                # self._batch_records_debug(
                #     _epoch, _batch_idx, _size_total_data, len(data), _size_batch, loss
                # )
                train_loss_lst.append(loss.item())
                _, predicted = torch.max(output.data, 1)
                epoch_total += target.size(0)
                epoch_correct += (predicted == target.data).sum().item()

            epoch_train_acc = epoch_correct / epoch_total * 100
            epoch_avg_loss = np.mean(train_loss_lst)

            # logger.debug(
            #     "client:{} round:{} epoch:{}/{} \t training loss:{:.4f} \t training acc:{:.2f}".format(
            #         client_id + 1,
            #         round_idx + 1,
            #         _epoch + 1,
            #         self.local_epochs,
            #         epoch_avg_loss,
            #         epoch_train_acc,
            #     )
            # )

            # fine-grained records
            # _, test_acc = self.model_evaluate(model, test_data_loader, criterion)
            # self.tensorboard.add_scalar(
            #     "{} - client {} Test Accuracy".format(self.fusion, client_id),
            #     test_acc,
            #     round_idx * self.local_epochs + _epoch,
            # )
            # self.tensorboard.flush()

        if (
            self.attacker_strategy == "model_poisoning_ipm"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            logger.info(f"client {client_id} is attacker, start poisoning model")
            crafted_model = ipm_attack_craft_model(
                self.server_model.to(self.device), model.to(self.device)
            )
            _, _test_acc = self.model_evaluate(
                crafted_model, test_data_loader, criterion
            )
            _train_loss, _ = self.model_evaluate(
                crafted_model, train_data_loader, criterion
            )
            return crafted_model, {"train_loss": _train_loss, "test_acc": _test_acc}
        elif (
            self.attacker_strategy == "model_poisoning_scaling"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            logger.info(f"client {client_id} is attacker, start poisoning model")
            crafted_model = scaling_attack(model.to(self.device))
            _, _test_acc = self.model_evaluate(
                crafted_model, test_data_loader, criterion
            )
            _train_loss, _ = self.model_evaluate(
                crafted_model, train_data_loader, criterion
            )
            return crafted_model, {"train_loss": _train_loss, "test_acc": _test_acc}
        elif (
            self.attacker_strategy == "model_poisoning_alie"
            and client_id in self.attacker_list
            and round_idx >= self.attack_start_round
        ):
            logger.info(f"client {client_id} is attacker, start poisoning model")
            crafted_model = alie_attack(model.to(self.device))
            _, _test_acc = self.model_evaluate(
                crafted_model, test_data_loader, criterion
            )
            _train_loss, _ = self.model_evaluate(
                crafted_model, train_data_loader, criterion
            )
            return crafted_model, {"train_loss": _train_loss, "test_acc": _test_acc}
        else:
            _test_loss, _test_acc = self.model_evaluate(
                model, test_data_loader, criterion
            )
            _train_loss, _train_acc = self.model_evaluate(
                model, train_data_loader, criterion
            )
            return model, {"train_loss": _train_loss, "test_acc": _test_acc}

    def _batch_records_debug(
        self,
        epoch: int,
        batch_idx: int,
        size_total_data: int,
        size_data: int,
        size_batch: int,
        loss: Any,
    ) -> None:
        if batch_idx % 10 == 0:
            logger.debug(
                "train epoch: {} [{}/{} ({:.0f}%)]\t training loss: {:.6f}".format(
                    epoch,
                    batch_idx * size_data,
                    size_total_data,
                    100.0 * batch_idx / size_batch,
                    loss.item(),
                )
            )

    def get_optimizer(self, model: nn.Module) -> optim.Optimizer:
        if self.optimizer == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.regularization,
            )
        elif self.optimizer == "amsgrad":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.regularization,
                amsgrad=True,
            )
        elif self.optimizer == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.regularization,
            )
        return optimizer

    def model_evaluate(
        self,
        model: nn.Module,
        data_loader: data.DataLoader,
        criterion: nn.CrossEntropyLoss,
    ) -> tuple:
        model.eval()

        loss = 0
        correct = 0
        with torch.no_grad():
            model.to(self.device)
            for _data, _targets in data_loader:
                data, targets = _data.to(self.device), _targets.to(self.device)
                outputs = model(data)
                loss += criterion(outputs, targets).item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == targets).sum().item()

        loss /= len(data_loader)
        accuracy = 100.0 * correct / len(data_loader.dataset)
        return loss, accuracy

    def aggregate_model(self, round_idx: int, model_updates: dict) -> Dict[str, Any]:
        logger.info("start model aggregation...fusion method: {}".format(self.fusion))

        if self.fusion == "average":
            average_params = fusion_avg(model_updates)
            return average_params
        elif self.fusion == "fedavg":
            data_sizes = {
                p_id: sum(len(batch[0]) for batch in self.client_data_loader[p_id][0])
                for p_id in self.round_client_list[round_idx]
            }
            logger.debug("data sizes: {}".format(data_sizes))
            weighted_avg_params = fusion_fedavg(model_updates, data_sizes)
            return weighted_avg_params
        elif self.fusion == "krum":
            # max_expected_adversaries = int(self.attacker_ratio * self.num_clients)
            max_expected_adversaries = int(self.attacker_ratio * len(model_updates))
            krum_params = fusion_krum(
                model_updates, max_expected_adversaries, self.device
            )
            return krum_params
        elif self.fusion == "median":
            median_params = fusion_median(model_updates, device=self.device)
            return median_params
        elif self.fusion == "clipping_median":
            median_clipping_params = fusion_clipping_median(
                model_updates, clipping_threshold=0.1, device=self.device
            )
            return median_clipping_params
        elif self.fusion == "trimmed_mean":
            trimmed_mean_params = fusion_trimmed_mean(
                model_updates, trimmed_ratio=0.1, device=self.device
            )
            return trimmed_mean_params
        elif self.fusion == "cos_defense":
            weighted_params = fusion_cos_defense(self.server_model, model_updates)
            return weighted_params
        elif self.fusion == "fools_gold":
            weighted_params = fusion_fools_gold(self.server_model, model_updates)
            return weighted_params
        elif self.fusion == "dual_defense":
            logger.info("start hyper-guard fusion with epsilon {}".format(self.epsilon))
            lst_round_attackers = intersection_of_lists(
                list(model_updates.keys()), self.attacker_list
            )
            logger.info(f"round {round_idx} attackers: {lst_round_attackers}")
            data_sizes = {
                p_id: sum(len(batch[0]) for batch in self.client_data_loader[p_id][0])
                for p_id in self.round_client_list[round_idx]
            }
            fused_params = fusion_dual_defense_enhance(
                self.server_model,
                model_updates,
                data_sizes,
                epsilon=self.epsilon,
            )
            return fused_params
        else:
            raise ValueError("Invalid fusion method")
