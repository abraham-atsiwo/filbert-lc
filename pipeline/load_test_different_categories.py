from typing import NamedTuple

from torchtext.data.utils import get_tokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from pipeline import read_csv_files_directory
from utils import set_seed

tokenizer = get_tokenizer("basic_english")


class TestDFTokenCategories:
    def __init__(
        self,
        directory,
        source,
        size_small,
        size_medium,
        size_large,
        same_len_per_class_small: bool = False,
        same_len_per_class_medium: bool = False,
        same_len_per_class_large: bool = False,
        seed: int = None,
    ) -> None:
        if seed:
            set_seed(seed)

        df = read_csv_files_directory(directory=directory, source=source)
        df["token_count"] = df["sentence"].apply(func=self._token_count)
        df["token_bucket"] = df["token_count"].apply(func=self._token_bucket)

        df_small = df[df["token_bucket"] == "small"]
        df_medium = df[df["token_bucket"] == "medium"]
        df_large = df[df["token_bucket"] == "large"]
        df_large = df_large.drop_duplicates(subset=[f"sentence_{j}" for j in range(5)])
        df_medium = df_medium.drop_duplicates(
            subset=[f"sentence_{j}" for j in range(3)]
        )
        df_small = df_small.drop_duplicates(subset=[f"sentence_{j}" for j in range(1)])

        df_small = self.sample_equal_categories(
            df_small, size_small, equal_len_per_class=same_len_per_class_small
        )
        df_medium = self.sample_equal_categories(
            df_medium, size_medium, equal_len_per_class=same_len_per_class_medium
        )
        df_large = self.sample_equal_categories(
            df_large, size_large, equal_len_per_class=same_len_per_class_large
        )

        self.test_small = df_small
        self.test_medium = df_medium
        self.test_large = df_large

    def sample_equal_categories(self, df, size, equal_len_per_class: bool = True):
        df_copy = df.copy()
        df_copy["label"] = np.int64(df_copy["label"])
        if not equal_len_per_class:
            if len(df_copy) < size:
                df_copy.sample(n=len(df_copy))
                return df_copy
            else:
                df_copy.sample(n=size)
                return df_copy[:size]

        else:
            output = []
            n_class = int((size / 3) + 1)
            for j in range(3):
                res = df[df["label"] == j]
                # print(len(res))
                output.append(res[:n_class])
            result = pd.concat(output)
            if len(result) >= size:
                result = result.sample(n=size, ignore_index=True)
            result.reset_index(inplace=True)
            return result

    def _token_count(self, text: str):
        size = tokenizer(text)
        return len(size)

    def _token_bucket(self, count: int):
        if count <= 150:
            return "small"
        elif count <= 300:
            return "medium"
        else:
            return "large"
