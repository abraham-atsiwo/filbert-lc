from typing import NamedTuple

from datasets import load_dataset
import pandas as pd 
from sklearn.model_selection import train_test_split
from allennlp.modules.elmo import Elmo
import torch.nn as nn
import torch

from lstm import lstm_elmo_trainer, predict_lstm_elmo, ElmoLSTM
from pipeline import TrainTestVal

# ELMo configuration
options_file = "elmo_options_wgt/elmo_2x1024_128_2048cnn_1xhighway_options.json"
weights_file = "elmo_options_wgt/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"


data = load_dataset("ab30atsiwo/sentences_50agree_gpt")


class TrainTest(NamedTuple):
    train: pd.DataFrame = None
    val: pd.DataFrame = None
    test: pd.DataFrame = None

data = TrainTestVal(
        gpt_path="data/gpt4",
        phrasebank_path="data/sentences_50agree",
        test_size=0.2,
        is_synthetic_in_test=False,
        trace=True,
        from_hub=False,
        seed=42,
    )

model = 'elmo'
model_path = f"../saved_models/lstm_phrasebank50agree_gpt/{model}"
params = {
    "train_data": data.train,
    "val_data": data.val,
    "test_data": data.test,
    "batch_size": 8,
    "options_file": options_file,
    "weights_file": weights_file,
    "embedding_dim": 512,
    "hidden_dim": 128,
    "output_dim": 3,
    "num_epochs": 5,
    "learning_rate": 0.001,
    "device": "mps",
    "is_less_better": False,
    "metric_for_best": "accuracy",
    "save_path": model_path,
    'trace': True
}

lstm_elmo_trainer(**params)





