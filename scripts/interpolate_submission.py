import argparse
import pandas as pd


def read_submission(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def interpolate(df: pd.DataFrame, frequency: int) -> pd.DataFrame:
    df = df.set_index(pd.Index(list(range(0, len(df.index) * frequency, frequency))))
    df = df.reindex(range(0, len(df.index) * frequency))
    df = df.interpolate()
    return df


def save_submission(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False)


def interpolate_submission(options) -> None:
    df = read_submission(options.input_path)
    df = interpolate(df, options.frequency)
    save_submission(df, options.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-path', '-i', type=str, required=True, help='Path where to read the input submission for processing'
    )
    parser.add_argument(
        '--output-path', '-o', type=str, required=True, help='Path where to save the output submission'
    )
    parser.add_argument(
        '--frequency', '-f', type=int, required=True,
        help='Integer sampling frequency used for dataset creation, e.g. 5, 10, 20'
    )

    args = parser.parse_args()
    interpolate_submission(args)
