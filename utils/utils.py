import random
from functools import wraps
import time
from typing import List, Dict

from transformers import AutoModelForSequenceClassification, BertModel
import torch
import matplotlib.pyplot as plt
import numpy as np




def detect_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        # torch.backends.mps.manual_seed(seed)
        pass


def token_count_distribution(df) -> None:
    def _count_tokens(text):
        return len(text.split())

    df["token_count"] = df["sentence"].apply(_count_tokens)
    plt.figure(figsize=(10, 6))
    plt.hist(
        df["token_count"],
        bins=range(df["token_count"].min(), df["token_count"].max() + 2),
        edgecolor="black",
    )
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.show()


def count_bert_parameters(
    model: BertModel = None, layer_num: List[int] = None, trace: bool = False
) -> Dict[str, int]:
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path="bert-base-uncased", num_labels=3
        )

    def helper(counter, parameter_dim):
        if num_par == 1:
            counter += parameter_dim[0]
        elif num_par == 2:
            counter += parameter_dim[0] * parameter_dim[1]
        return counter

    layer_count = 0
    embed_count = 0
    output_count = 0
    for name, parameter in model.named_parameters():
        name = name.split(".")
        parameter_dim = parameter.shape
        num_par = len(parameter_dim)

        if name[1] == "embeddings":
            embed_count = helper(embed_count, parameter_dim)
        elif len(name) >= 4 and name[3] in [str(j) for j in layer_num]:
            layer_count = helper(layer_count, parameter_dim)
        elif name[0] == "classifier" or name[1] == "pooler":
            output_count = helper(output_count, parameter_dim)
        # else:
        #     print(name)
        if trace:
            print((name, parameter_dim))

    total = sum([output_count, layer_count, embed_count])
    return {
        "total": total,
        "embedding": embed_count,
        "layer": layer_count,
        "output": output_count,
    }
