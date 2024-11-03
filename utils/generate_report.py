from typing import Union, Dict
import os

import pandas as pd
import torch


from finbert_gpt import predict_finbertgpt
from utils import detect_device
from lstm import predict_lstm_fasttext_bert_glove, predict_lstm_elmo


# ELMo configuration
options_file = "elmo_options_wgt/elmo_2x1024_128_2048cnn_1xhighway_options.json"
weights_file = "elmo_options_wgt/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"


def generate_report(
    model_name: str = None,
    test_df: pd.DataFrame = None,
    device: str = None,
    batch_size: int = None,
    is_lstm_type: bool = False,
    id2label: Dict[int, str] = None,
    trace: bool = True,
) -> Union[Dict[str, pd.DataFrame], None]:
    if id2label is None:
        id2label = {0: "negative", 1: "neutral", 2: "positive"}
    if device is None:
        device = detect_device()
    if batch_size is None:
        batch_size = 1
    if model_name is None:
        if not is_lstm_type:
            model_name = "bert-base-uncased"
        else:
            model_name = "glove"
    if not isinstance(test_df, pd.DataFrame):
        raise ValueError("Test Data must be a Dataframe")
    else:
        columns = list(test_df.columns)
        if not ("sentence" in columns or "label" in columns):
            raise ValueError("Dataframe must have sentence and label columns")

    if trace:
        print(f"Using model_name: {model_name}")
        print(f"Using device: {device}")

    # load model and tokenizer from pretrained
    if not is_lstm_type:
        result = predict_finbertgpt(
            model_name=model_name,
            test_df=test_df,
            device=device,
            test_id2label=id2label,
        )
    else:
        path = model_name.split("/")[-1]
        if trace:
            print(path)
        if path.lower() == "elmo":
            result = predict_lstm_elmo(
                model_path=model_name,
                df=test_df,
                device=device,
                options_file=options_file,
                weights_file=weights_file,
            )
        elif path.lower() in ["bert", "glove", "fasttext"]:
            result = predict_lstm_fasttext_bert_glove(
                df=test_df,
                device=device,
                criterion=torch.nn.CrossEntropyLoss,
                model_path=model_name,
            )
    return result
