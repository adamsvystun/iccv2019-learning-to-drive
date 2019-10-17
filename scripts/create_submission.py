import argparse
import math

import pandas as pd


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
    initial_offset = sample_frequency * sequence_length
    # For each chapter
    for chapter in original_dataset_descriptor.chapter.unique():
        # Get chapters
        original_chapter = original_dataset_descriptor[original_dataset_descriptor.chapter == chapter]
        original_chapter_len = len(original_chapter.index)
        sampled_chapter = sampled_dataset_descriptor[sampled_dataset_descriptor.chapter == chapter]
        sampled_chapter_len = len(sampled_chapter.index)
        # Get relevant submission cut
        submission_cut = submission.head(sampled_chapter_len - sequence_length + 1)
        submission = submission.drop(submission_cut.index)
        # Interpolation submission cut
        old_indices = list(range(initial_offset, (len(submission_cut.index) + sequence_length) * sample_frequency, sample_frequency))
        submission_cut = submission_cut.set_index(pd.Index(old_indices))
        new_indices = range(0, original_chapter_len)
        # Start from 100s frame in the test data
        new_indices = new_indices[100:original_chapter_len]
        submission_cut = submission_cut.reindex(new_indices)
        # Interpolate (replace NaN's) in the data frame in both directions
        submission_cut = submission_cut.interpolate(limit_direction='both')
        # Append new submission to a list
        interpolated_submissions.append(submission_cut)
    # Return concatenated data frame
    return pd.concat(interpolated_submissions, axis=0)


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
        options.number
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
        '--number', '-n', type=int, required=True,
        help='What number of previous frames has been loaded for test prediction'
    )

    args = parser.parse_args()
    create_submission(args)
