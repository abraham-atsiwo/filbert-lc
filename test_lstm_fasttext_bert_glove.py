from typing import NamedTuple

from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn

from lstm import lstm_fasttext_bert_glove, predict_lstm_fasttext_bert_glove
from utils import set_seed
from pipeline import TrainTestVal

set_seed(42)


data = TrainTestVal(
        gpt_path="data/gpt4",
        phrasebank_path="data/sentences_50agree",
        test_size=0.2,
        is_synthetic_in_test=False,
        trace=True,
        from_hub=False,
        seed=42,
    )

model = 'fasttext'
# Example of how to call the function with parameters
save_path = f"../saved_models/lstm_phrasebank50agree_gpt/{model}"
params = {
    "epoch": 6,
    "train_val_test_df": data,
    "embedding_option": f"{model}",
    "glove_dim": 300,
    "device": "mps",
    "batch_size": 8,
    "hidden_size": 128,
    "output_size": 3,
    "criterion": nn.CrossEntropyLoss,
    "optimizer": torch.optim.Adam,
    "learning_rate": 0.001,
    "save_path": save_path,
    "metric_for_best_model": "accuracy",
    "is_lower_better": False,
    "trace": True,
    "seed": 42,
    "freeze": True,
}
# lstm_fasttext_bert_glove(**params)


# # # Call the function with the parameters unpacked from the dictionary
predict_lstm_fasttext_bert_glove(
    df=data.test, criterion=nn.CrossEntropyLoss, device="mps", model_path=save_path
)
