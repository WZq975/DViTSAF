import argparse

def get_config_parser():
    parser = argparse.ArgumentParser(description="Let's distill vision transformers.")

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--batch_size", type=int, default=256, help="batch size (default: %(default)s)."
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--objective",
        type=str,
        choices=["teacher", "student", "distillation"],
        default="distillation",
        help="train teacher or student, or do distillation (default: %(default)s).",
    )
    # model.add_argument(
    #     "--teacher",
    #     type=str,
    #     choices=['vit-large', 'vit-base'],
    #     default='vit-large',
    #     help='use vit-base or vit-large as the teacher model (default: %(default)s).'
    # )
    model.add_argument(
        "--distillation",
        type=str,
        choices=['KD', 'SAF', 'SAF+KD'],
        default='SAF',
        help='distillation method: KD, SAF(ours), SAF+KD (default: %(default)s).'
    )

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="number of epochs for training (default: %(default)s).",
    )
    # optimization.add_argument(
    #     "--optimizer",
    #     type=str,
    #     default="adam",
    #     choices=["sgd", "momentum", "adam", "adamw"],
    #     help="choice of optimizer (default: %(default)s).",
    # )
    optimization.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate for Adam optimizer (default: %(default)s).",
    )
    # optimization.add_argument(
    #     "--momentum",
    #     type=float,
    #     default=0.9,
    #     help="momentum for SGD optimizer (default: %(default)s).",
    # )
    # optimization.add_argument(
    #     "--weight_decay",
    #     type=float,
    #     default=5e-4,
    #     help="weight decay (default: %(default)s).",
    # )

    dataset = parser.add_argument_group("Dataset")
    dataset.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=["CIFAR-10", "CIFAR-100", "Flowers-102"],
        help="Name of dataset (default: %(default)s).",
    )

    exp = parser.add_argument_group("Experiment config")
    exp.add_argument(
        "--logdir",
        type=str,
        default='./logs',
        help="unique experiment identifier (default: %(default)s).",
    )
    # exp.add_argument(
    #     "--seed",
    #     type=int,
    #     default=42,
    #     help="random seed for repeatability (default: %(default)s).",
    # )

    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda",
        help="device to store tensors on (default: %(default)s).",
    )
    # misc.add_argument(
    #     "--print_every",
    #     type=int,
    #     default=100,
    #     help="number of minibatches after which to print loss (default: %(default)s).",
    # )
    # misc.add_argument(
    #     "--visualize",
    #     action='store_true',
    #     help='A flag to visualize the filters or MLP layer at the end'
    # )
    return parser
