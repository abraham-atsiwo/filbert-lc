from typing import List, Dict 
import os

from datasets import load_dataset
import pandas as pd


def load_data(dataset_path):
    try:
        dataset = load_dataset("csv", data_files=dataset_path)
        print("Dataset loaded successfully from the given path.")
        return dataset
    except Exception as e:
        print(f"Failed to load dataset from path: {e}")
        try:
            dataset_name = os.path.basename(dataset_path)
            print(f"Searching for dataset: {dataset_name}")
            dataset = load_dataset(dataset_name)
            print("Dataset loaded successfully from Hugging Face.")
            return dataset
        except Exception as e:
            print(f"Failed to load dataset from Hugging Face: {e}")
            return None


def load_remote_df(path: List[str], id2label: Dict[int, str]):
    pass
