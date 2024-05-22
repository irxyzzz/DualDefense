import os
import datetime

from args import args_parser

if __name__ == "__main__":
    args = args_parser()
    print(f"args: {args}")

    config = {
        "num_clients": args.num_clients,
        "dataset": args.dataset,
        "fusion": args.fusion,
        "training_round": args.training_round,
        "local_epochs": args.local_epochs,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "data_dir": args.dir_data,
        "partition_type": args.partition_type,
        "partition_dirichlet_beta": args.partition_dirichlet_beta,
        "regularization": args.regularization,
        "attacker_ratio": args.attacker_ratio,
        "attacker_strategy": args.attacker_strategy,
        "attack_start_round": args.attack_start_round,
        "epsilon": args.epsilon,
        "device": args.device,
        "seed": args.seed,
    }
    print(f"config: {config}")

    log_dir = "./log"
    tensorboard_dir = "./log"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    log_file = "{}-{}-{}-p{}r{}e{}b{}ar{}as{}-epsilon{}-{}.log".format(
        args.dataset,
        args.fusion,
        args.attacker_strategy,
        args.num_clients,
        args.training_round,
        args.local_epochs,
        args.batch_size,
        args.attacker_ratio,
        args.attack_start_round,
        args.epsilon,
        timestamp,
    )
    os.environ["LOG_FILE_NAME"] = os.path.join(log_dir, log_file)

    from utils.util_logger import setup_tensorboard

    tb_file = "{}-{}-{}-p{}r{}e{}b{}ar{}as{}-epsilon{}-{}".format(
        args.dataset,
        args.fusion,
        args.attacker_strategy,
        args.num_clients,
        args.training_round,
        args.local_epochs,
        args.batch_size,
        args.attacker_ratio,
        args.attack_start_round,
        args.epsilon,
        timestamp,
    )

    config["tensorboard"] = setup_tensorboard(tensorboard_dir, tb_file)

    from fl import SimulationFL

    fl = SimulationFL(config)
    fl.start()
