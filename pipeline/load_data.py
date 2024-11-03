from typing import List
import os
import random

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torchtext.data import get_tokenizer
from datasets import load_dataset, DatasetDict

from utils import set_seed
from .utils import read_csv_files_directory


class TrainTestVal:

    def __init__(
        self,
        gpt_path: str = None,
        concatenate_path: str = None,
        phrasebank_path: str = None,
        test_size: int = None,
        is_synthetic_in_test: bool = False,
        trace: bool = False,
        from_hub: bool = False,
        train_test_val_from_hub: bool = False,
        seed: int = None,
    ) -> None:
        self.seed = None
        if seed:
            set_seed(seed)
        self.train = None
        self.test = None
        self.val = None

        if train_test_val_from_hub:
            if is_synthetic_in_test:
                data = load_dataset("ab30atsiwo/finbert_gpt_syn_in_test")
            else:
                data = load_dataset("ab30atsiwo/finbert_gpt")
            self.train = data["train"]
            self.test = data["test"]
            self.val = data["validation"]
            return
        columns = ["sentence", "label", "comb_size", "source"]
        df = pd.DataFrame([], columns=columns)

        if concatenate_path:
            df_concat = read_csv_files_directory(
                concatenate_path, is_concat=True, source="concat_phrasebank_train"
            )[columns]
            df = pd.concat((df, df_concat))

        if phrasebank_path:
            if from_hub:
                phrasebank = load_dataset(
                    "financial_phrasebank", phrasebank_path, trust_remote_code=True
                )["train"].to_pandas()
                if seed:
                    train, tmp = train_test_split(
                        phrasebank,
                        test_size=test_size,
                        stratify=phrasebank["label"],
                        random_state=seed,
                        shuffle=True,
                    )
                    test, val = train_test_split(
                        tmp,
                        test_size=0.5,
                        stratify=tmp["label"],
                        random_state=seed,
                        shuffle=True,
                    )
                else:
                    train, tmp = train_test_split(
                        phrasebank,
                        test_size=test_size,
                        stratify=phrasebank["label"],
                        shuffle=True,
                    )
                    test, val = train_test_split(
                        tmp, test_size=0.5, stratify=tmp["label"], shuffle=True
                    )
            else:
                train = pd.read_csv(f"{phrasebank_path}/train.csv")
                test = pd.read_csv(f"{phrasebank_path}/test.csv")
                val = pd.read_csv(f"{phrasebank_path}/val.csv")
            train["source"] = "phrasebank"
            train["comb_size"] = 1
            test["source"] = "phrasebank"
            test["comb_size"] = 1
            val["source"] = "phrasebank"
            val["comb_size"] = 1

        if is_synthetic_in_test:
            df = pd.concat((df, train, test, val))
            df.drop_duplicates(subset=["sentence"], inplace=True)
            if seed:
                train, tmp = train_test_split(
                    df,
                    test_size=test_size,
                    stratify=df["label"],
                    random_state=seed,
                    shuffle=True,
                )
                test, val = train_test_split(
                    tmp,
                    test_size=0.5,
                    stratify=tmp["label"],
                    random_state=seed,
                    shuffle=True,
                )
            else:
                train, tmp = train_test_split(
                    df, test_size=test_size, stratify=df["label"], shuffle=True
                )
                test, val = train_test_split(
                    tmp, test_size=0.5, stratify=tmp["label"], shuffle=True
                )
        else:
            train = pd.concat((train, df))

        if gpt_path:
            df_gpt = read_csv_files_directory(
                directory=gpt_path, is_concat=False, source="gpt4"
            )[columns]
            train = pd.concat((train, df_gpt))

        train.drop_duplicates(subset=["sentence"], inplace=True)
        test.drop_duplicates(subset=["sentence"], inplace=True)
        val.drop_duplicates(subset=["sentence"], inplace=True)

        self.train = train
        self.test = test
        self.val = val

        if trace:
            print(train.shape)
            print(test.shape)
            print(val.shape)


def _get_token_count(sentence):
    tokenizer = get_tokenizer("basic_english")
    return len(tokenizer(sentence))


def _token_bucket(token_count):
    if token_count <= 80:
        return "small"
    if token_count <= 200:
        return "medium"
    return "large"


