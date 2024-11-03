import argparse
import os

import pandas as pd



def read_csv_files_directory(
    directory: str = None, is_concat: bool = False, source: str = None, columns=None
):
    if columns is None:
        df = pd.DataFrame([], columns=["sentence", "label", "comb_size"])
    else:
        df = pd.DataFrame([], columns=columns)
    all_files = os.listdir(directory)
    csv_files = [f for f in all_files if f.endswith(".csv")]
    for file in csv_files:
        file = os.path.join(directory, file)
        tmp_df = pd.read_csv(filepath_or_buffer=file)
        if is_concat:
            tmp_df.drop_duplicates(subset=["sentence"], inplace=True)
        df = pd.concat((df, tmp_df))
    df["source"] = source
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        help="Path to a datafile!",
        default="sentences_50agree/test.csv",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        default="cpu",
        help="A device to train the model on: mps/cpu/cuda!",
    )
    parser.add_argument(
        "--concatenate",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help="Whether to consider the history for isNSP.",
        required=False,
    )
    parser.add_argument(
        "--save_filename",
        type=str,
        required=False,
        help="path where the generated output should be saved!",
        default="sentences_50agree_train_4.csv",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=0,
        help="A numeric representation of the label column!",
        required=False,
    )
    parser.add_argument(
        "--num_comb",
        type=int,
        default=2,
        help="The number of combination to generate!",
        required=False,
    )
    parser.add_argument(
        "--n_core", type=int, default=2, help="Number of cores", required=False
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="An optional string argument",
        required=False,
        default="../saved_models/lambda-labs-nsp-200000",
    )
    parser.add_argument(
        "--is_terminal_args",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help="Whether to consider the history for isNSP.",
        required=False,
    )
    return parser.parse_args()
