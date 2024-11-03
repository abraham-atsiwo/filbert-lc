import pandas as pd
import numpy as np 

from utils import set_seed
from pipeline import TrainTestVal
from utils import token_count_distribution


set_seed(42)

data = TrainTestVal(
    concatenate_path='data/concat_data_history',
    gpt_path="data/gpt4",
    phrasebank_path="data/sentences_50agree",
    test_size=0.2,
    is_synthetic_in_test=False,
    trace=True,
    from_hub=False,
    seed=42,
)

token_count_distribution(df=data.train)
token_max = np.max(data.train['token_count'])
print(token_max)