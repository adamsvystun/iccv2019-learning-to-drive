import argparse
import math

import matplotlib
from scipy.interpolate import UnivariateSpline
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

matplotlib.rcParams['figure.dpi'] = 200
max_order = 1
window_size = 5
extrapolation_type = 'spline'      # [spline, curve]
spline_k = 1
plot_first = False


def read_descriptor(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def read_submission(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def infer_frequency(
        original_dataset_descriptor: pd.DataFrame,
        sampled_dataset_descriptor: pd.DataFrame
) -> int:
    """Infer the sampling frequency of the dataset, so we don't ask the user for redundant information"""
    original_chapter_one = original_dataset_descriptor[original_dataset_descriptor.chapter == 0]
    sampled_chapter_one = sampled_dataset_descriptor[sampled_dataset_descriptor.chapter == 0]
    return math.floor(float(len(original_chapter_one.index)) / len(sampled_chapter_one.index))


def get_function_to_fit(n, max_order=max_order):
    # Function to curve fit to the data
    def function_to_fit(x, *args):
        f = 0
        for i in range(min(n, max_order + 1)):
            f += args[i] * np.power(x, i)
        return f
    return function_to_fit


# def get_function_to_fit(n):
#     if n == 1:
#         return lambda x, a: a
#     elif n == 2:
#         return lambda x, a, b: x * b + a
#     elif n == 3:
#         return lambda x, a, b, c: x ** 2 * c + x * b + a
#     else:
#         return lambda x, a, b, c, d: x ** 3 * d + x ** 2 * c + x * b + a


def interpolate(
        submission: pd.DataFrame,
        original_dataset_descriptor: pd.DataFrame,
        sampled_dataset_descriptor: pd.DataFrame,
        sample_frequency: int,
        frequency: int,
        number: int
) -> pd.DataFrame:
    interpolated_submissions = []
    sequence_length = frequency * number
    # Calculate the initial offset of the first predicted frame under given configuration
    initial_offset = sample_frequency * (sequence_length + 1)
    our_original_submission = pd.read_csv('data/sub_val_2.csv')
    # For each chapter
    for chapter in original_dataset_descriptor.chapter.unique():
        # Get chapters
        original_chapter = original_dataset_descriptor[original_dataset_descriptor.chapter == chapter]
        original_chapter_len = len(original_chapter.index)
        sampled_chapter = sampled_dataset_descriptor[sampled_dataset_descriptor.chapter == chapter]
        sampled_chapter_len = len(sampled_chapter.index)
        # Get relevant submission cut
        submission_cut = submission.head(sampled_chapter_len - sequence_length)
        submission = submission.drop(submission_cut.index)
        old_indices = list(range(
            initial_offset, (len(submission_cut.index) + sequence_length + 1) * sample_frequency, sample_frequency
        ))
        submission_cut = submission_cut.set_index(pd.Index(old_indices))
        new_indices = range(0, original_chapter_len)
        new_indices = new_indices[100:original_chapter_len]
        extrapolated_cut = pd.DataFrame(index=pd.Index(new_indices), columns=['canSpeed'])
        mean = 13.426
        for i in range(100, old_indices[0]):
            extrapolated_cut['canSpeed'][i] = mean
        for index, row in submission_cut.iterrows():
            # Interpolation submission cut
            history_indices = [i for i in old_indices if i <= index]
            history = submission_cut[submission_cut.index <= index]
            window_history_indices = history_indices[-(window_size):]
            window_history = history[-(window_size):]

            if len(old_indices) < len(history_indices):
                max_index = old_indices[len(history_indices)]
            else:
                max_index = new_indices[-1] + 1
            new_indices_step = np.array([i for i in new_indices if index < i < max_index])

            extrapolated_cut['canSpeed'][index] = row.canSpeed
            if len(window_history_indices) <= spline_k or extrapolation_type == 'curve':
                guess = [0.5] * min(len(history_indices), (max_order+1))
                function_to_fit = get_function_to_fit(len(history_indices))
                params, _ = curve_fit(
                    function_to_fit, window_history_indices, window_history.canSpeed.values, guess
                )

                extrapolated_cut['canSpeed'][new_indices_step] = np.maximum(0, function_to_fit(new_indices_step, *params))
            else:
                extrapolator = UnivariateSpline(window_history_indices, window_history.canSpeed.values, k=spline_k)
                extrapolated_cut['canSpeed'][new_indices_step] = np.maximum(0, extrapolator(new_indices_step))
        #
        # # Interpolate (replace NaN's) in the data frame in both directions
        # submission_cut = submission_cut.interpolate(
        #     limit_direction='both',
        #     method='linear',
        # )
        # Append new submission to a list
        if plot_first:
            print(extrapolated_cut.canSpeed.values.shape)
            print(our_original_submission.iloc[np.arange(len(extrapolated_cut.index))].canSpeed.values.shape)
            plt.plot(extrapolated_cut.canSpeed.values)
            plt.plot(our_original_submission.iloc[np.arange(len(extrapolated_cut.index))].canSpeed.values)
            plt.show()
            exit()
        interpolated_submissions.append(extrapolated_cut)
    # Return concatenated data frame
    original_dataset_descriptor = original_dataset_descriptor.groupby('chapter').apply(lambda x: x.iloc[100:])
    our_submission = pd.concat(interpolated_submissions, axis=0)
    mse = mean_squared_error(original_dataset_descriptor.canSpeed.values, our_submission.canSpeed.values)
    print(mse)
    return our_submission


def save_submission(df: pd.DataFrame, filepath: str) -> None:
    df.to_csv(filepath, index=False)


def create_submission(options) -> None:
    original_dataset_descriptor = read_descriptor(options.original_dataset_path)
    sampled_dataset_descriptor = read_descriptor(options.sampled_dataset_path)
    submission = read_submission(options.input_path)
    sample_frequency = infer_frequency(original_dataset_descriptor, sampled_dataset_descriptor)
    submission = interpolate(
        submission,
        original_dataset_descriptor,
        sampled_dataset_descriptor,
        sample_frequency,
        options.frequency,
        options.sequence_frame_number
    )
    save_submission(submission, options.output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-path', '-i', type=str, required=True, help='Path where to read the input submission for processing'
    )
    parser.add_argument(
        '--output-path', '-o', type=str, required=True, help='Path where to save the output submission'
    )
    parser.add_argument(
        '--original-dataset-path', '-od', type=str, required=True,
        help='Path to original dataset csv describing frames'
    )
    parser.add_argument(
        '--sampled-dataset-path', '-sd', type=str, required=True,
        help='Path to sampled dataset csv describing frames'
    )
    parser.add_argument(
        '--frequency', '-f', type=int, required=True, default=1,
        help='With what frequency the dataset has been loaded for test prediction'
    )
    parser.add_argument(
        '--sequence-frame-number', '-sfn', type=int, required=True,
        help='What number of previous frames has been loaded for test prediction'
    )

    args = parser.parse_args()
    create_submission(args)
