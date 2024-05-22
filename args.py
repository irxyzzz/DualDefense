from __future__ import print_function

import argparse


def args_parser():
    parser = argparse.ArgumentParser(
        description="a simulation of federated learning with defense mechanisms."
    )

    # federated settings
    parser.add_argument(
        "-np",
        "--num_clients",
        type=int,
        default=5,
        help="number of workers in a distributed cluster",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fmnist", "cifar10", "svhn"],
        help="dataset used for training",
    )
    parser.add_argument(
        "-pd",
        "--partition_type",
        type=str,
        default="noniid",
        help="data partitioning strategy",
    )
    parser.add_argument(
        "-pb",
        "--partition_dirichlet_beta",
        type=float,
        default=0.25,
        help="dirichlet distribution parameter for data partitioning",
    )
    parser.add_argument(
        "-f",
        "--fusion",
        choices=[
            "average",
            "fedavg",
            "krum",
            "median",
            "clipping_median",
            "trimmed_mean",
            "cos_defense",
            "dual_defense",
        ],
        type=str,
        default=0.5,
        help="dirichlet distribution parameter for data partitioning",
    )
    parser.add_argument(
        "-dm",
        "--dir_model",
        type=str,
        required=False,
        default="./models/",
        help="Model directory path",
    )
    parser.add_argument(
        "-dd",
        "--dir_data",
        type=str,
        required=False,
        default="./data/",
        help="Data directory",
    )

    # hyperparameters settings
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.01,
        help="learning rate (default: 0.1)",
    )
    parser.add_argument(
        "-le", "--local_epochs", type=int, default=1, help="number of local epochs"
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training",
    )
    parser.add_argument(
        "-tr",
        "--training_round",
        type=int,
        default=100,
        help="number of maximum communication roun",
    )
    parser.add_argument(
        "-re",
        "--regularization",
        type=float,
        default=1e-5,
        help="L2 regularization strength",
    )
    parser.add_argument(
        "-op",
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "amsgrad"],
        help="optimizer used in the training process",
    )
    # adv and defense settings
    parser.add_argument(
        "--attacker_ratio",
        type=float,
        default=0.2,
        required=False,
        help="ratio for number of attackers",
    )
    parser.add_argument(
        "--attacker_strategy",
        type=str,
        default="none",
        required=False,
        choices=[
            "none",
            "model_poisoning_ipm",
            "model_poisoning_scaling",
            "model_poisoning_alie",
        ],
        help="attacker strategy",
    )
    parser.add_argument(
        "--attack_start_round",
        type=int,
        default=-1,
        required=False,
        help="the round to start attack",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        required=False,
        help="differential privacy epsilon used by dual_defense method",
    )
    # parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label')
    # parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path')
    # parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size')

    # other settings
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        required=False,
        choices=["cpu", "mps", "cuda"],
        help="device to run the program with pytorch",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7890,
        required=False,
        help="Random seed",
    )

    return parser.parse_args()
