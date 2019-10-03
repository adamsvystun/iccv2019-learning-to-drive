import argparse

from src.options import get_default_parser


def evaluate(options):
    pass


if __name__ == '__main__':
    parser = get_default_parser()

    # parser.add_argument(
    #     '--dataset-path', type=str, default="/data/comixify/datasets/rl/eccv16_dataset_summe_google_pool5.h5"
    # )
    # parser.add_argument('--pre-lstm-dropout', type=float, default=None)

    args = parser.parse_args()
    evaluate(args)
