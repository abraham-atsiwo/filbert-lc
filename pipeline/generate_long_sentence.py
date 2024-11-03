import random

from multiprocessing import Pool
import multiprocessing
from typing import List, Union
import os
import argparse
import csv
from itertools import combinations
from functools import partial

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def generate_combination(data, comb_size):
    return combinations(data, comb_size)


class GenerateLongSequence:

    def __init__(self, model_name: str, seed: int = None) -> None:
        # if seed:
        #     set_seed(seed)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def __call__(
        self,
        sentences: List,
        device: str,
        concatenate: bool = False,
        filename: str = None,
        label: Union[int, str] = None,
    ):
        self.predict_multiple_nsp(
            sentences=sentences,
            device=device,
            concatenate=concatenate,
            filename=filename,
            label=label,
        )

    def predict_one_nsp(self, sentenceA, sentenceB, device):
        inputs = self.tokenizer.encode_plus(sentenceA, sentenceB, return_tensors="pt")
        inputs.to(device=device)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        return probs[0, 1].item()

    def predict_multiple_nsp(
        self,
        sentences: List,
        device: str,
        concatenate: bool = False,
        filename: str = None,
        label: Union[int, str] = None,
    ):
        valid = True
        for i in range(len(sentences) - 1):
            if concatenate:
                sentenceA = "".join(sentences[: i + 1])
                sentenceB = sentences[i + 1]
            else:
                sentenceA = sentences[i]
                sentenceB = sentences[i + 1]
            if self.predict_one_nsp(sentenceA, sentenceB, device=device) <= 0.5:
                valid = False
                break
        if filename is None:
            return valid, sentences, label
        else:
            if valid:
                file_exists = os.path.isfile(filename)
                result = "".join(sentences)
                row = list(sentences) + [result, label, len(sentences), valid, len(result.split())]

                with open(f"{filename}", mode="a", newline="") as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow(
                            [f"sentence_{num}" for num in range(len(sentences))]
                            + ["sentence", "label", "comb_size", "valid_sequence", "token_len"]
                        )
                    writer.writerow(row)


def generate_long_sentence(
    data: pd.DataFrame = None,
    device: str = None,
    concatenate: bool = None,
    save_filename: str = None,
    label: Union[int, str] = None,
    num_comb: int = None,
    data_path: str = None,
    model_name: str = None,
    n_core: int = None,
    is_terminal_args: bool = False,
    concatenate_path: str = None, 
) -> None:
    # comb size
    numbers = [1000, 150, 20, 10, 10, 10, 20, 20, 20]
    combination_size = {str(j + 2): num for j, num in enumerate(numbers)}
    print(combination_size)
    n_core = n_core if n_core is not None else multiprocessing.cpu_count() - 1
    if data_path is not None:
        df = pd.read_csv(data_path)
    else:
        df = data
    if not concatenate:
        # name = "concat_data"
        name = concatenate_path
    else:
        # name = "concat_data_history"
        name = concatenate_path
    data = df[df["label"] == label]["sentence"].to_list()
    random.shuffle(data)
    data = data[: combination_size[str(num_comb)]]
    # print(len(data))
    if not os.path.exists(name):
        os.makedirs(name)
    if save_filename is None:
        filename = f"{name}/{data_path}"
    else:
        filename = f"{name}/{save_filename}"
    comb_data = list(generate_combination(data, num_comb))
    random.shuffle(comb_data)
    model = GenerateLongSequence(model_name=model_name)

    # print(n_core)
    # generate combination in parallel
    with Pool(processes=n_core) as pool:
        # Use partial to fix the offset parameter
        partial_square = partial(
            model.predict_multiple_nsp,
            device=device,
            concatenate=concatenate,
            filename=filename,
            label=label,
        )
        # Map the partial_square function to the iterator of numbers
        results = pool.map(partial_square, comb_data)





