import os
import random
import re
import multiprocessing
from multiprocessing import Pool
from functools import partial
from typing import List

import pandas as pd
import random

from utils import timer_decorator, set_seed


@timer_decorator
def create_nsp_labels(path, size: int, seed: int = None):
    if seed:
        set_seed(seed)
    extension = path.split(".")[-1]
    if extension == "parquet":
        df = pd.read_parquet(path)
    elif extension == "csv":
        df = pd.read_csv(path)
    df = df.sample(n=size)
    df["label"] = 1
    df_not_nsp = []
    for j in range(size):
        indB = random.randint(0, size - 1)
        while indB == j:
            indB = random.randint(0, size - 1)
        sentenceA = df.iloc[j]["sentenceA"]
        sentenceB = df.iloc[indB]["sentenceB"]
        row = [sentenceA, sentenceB, 0]
        df_not_nsp.append(row)
    df_not_nsp = pd.DataFrame(df_not_nsp)
    df_not_nsp.columns = ["sentenceA", "sentenceB", "label"]
    df = pd.concat((df_not_nsp, df), ignore_index=True)
    return df.sample(n=len(df))


def split_bloombery_news(path: str, start: int, end: int):
    try:
        with open(
            path,
            "r",
        ) as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines if line.strip()][start:end]
        text = "".join(lines)
        pattern = re.compile(
            r"(?<!\b[A-Z])(?<!\b[A-Z]\.)(?<!\b[A-Z]\.[A-Z])(?<!\d)\. +(?=[A-Z])"
        )
        sentences = pattern.split(text)
        df = {}
        df["sentenceA"] = sentences[:-1]
        df["sentenceB"] = sentences[1:]
        return pd.DataFrame(df)
    except:
        df = {}
        df["sentenceA"] = []
        df["sentenceB"] = []
        return pd.DataFrame(df)


def concatenate_multiple_files(multiple_path: List[str], start: int, end: int):
    isNext = []
    for path in multiple_path:
        try:
            # print(path)
            isNext.append(split_bloombery_news(path=path, start=start, end=end))
        except:
            df = {}
            df["sentenceA"] = []
            df["sentenceB"] = []
            isNext.append(pd.DataFrame(df))
    isNext = pd.concat(isNext, ignore_index=True)
    return isNext


def read_all_files_in_nested_folders(root_directory, file_extension="*"):
    path = []
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            path.append(file_path)
    return path


@timer_decorator
def parallel_split_bloombery_news(func, data, num_workers=None, *args, **kwargs):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    func = partial(func, *args, **kwargs)
    with Pool(processes=num_workers) as pool:
        results = pool.map(func, data)
    return results

@timer_decorator
def generate_test_from_bloombery(root_directory, save_path):
    file_paths = read_all_files_in_nested_folders(root_directory)
    results = parallel_split_bloombery_news(
        split_bloombery_news, file_paths, start=10, end=-10
    )
    df = pd.concat(results, ignore_index=True)
    df.to_csv()
    print(len(df))

