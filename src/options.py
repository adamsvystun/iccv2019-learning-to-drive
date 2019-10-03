import argparse


def get_default_parser():
    parser = argparse.ArgumentParser()

    # Data paths
    parser.add_argument(
        '--dataset-path', type=str, default="/data/comixify/datasets/rl/eccv16_dataset_summe_google_pool5.h5"
    )
    parser.add_argument('--augment-dataset-path', nargs='+', required=False)

    # Dataset parameters
    parser.add_argument('--dataset-augment', type=bool, default=False)
    parser.add_argument('--dataset-augment-pad', type=int, default=90)
    parser.add_argument('--dataset-augment-len', type=int, default=300)

    # Train parameters
    parser.add_argument('--cross-val', default=False)
    parser.add_argument('--features-size', default=1024)
    parser.add_argument('--epochs', default=325)

    # Optimizer
    parser.add_argument('--decay', default=0.001)
    parser.add_argument('--layer-normalization', default=True)

    # Pre-LSTM network
    parser.add_argument('--pre-lstm-net', nargs='+', default=[256, 128, 32])
    parser.add_argument('--pre-lstm-dropout', type=float, default=None)

    # LSTM
    parser.add_argument('--hidden-dim', nargs='+', default=[8, 4])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--recurrent-dropout', type=float, default=0.5)

    # Regularizer
    parser.add_argument('--kernel-regularizer', type=str, default=None)
    parser.add_argument('--recurrent-regularizer', type=str, default=None)
    parser.add_argument('--bias-regularizer', type=str, default=None)
    parser.add_argument('--activity-regularizer', type=str, default=None)

    # Output dense
    parser.add_argument('--output-dense-dropout', type=float, default=None, help="")
    return parser
