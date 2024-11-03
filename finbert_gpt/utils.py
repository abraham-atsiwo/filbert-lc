from functools import wraps
import time
import random

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)


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


def compute_metrics(eval_pred):
    p = eval_pred
    preds = np.argmax(p.predictions, axis=1)
    accuracy = accuracy_score(p.label_ids, preds)
    f1_micro = f1_score(p.label_ids, preds, average="micro")
    f1_macro = f1_score(p.label_ids, preds, average="macro")
    precision_micro = precision_score(p.label_ids, preds, average="micro")
    precision_macro = precision_score(p.label_ids, preds, average="macro")
    recall_micro = recall_score(p.label_ids, preds, average="micro")
    recall_macro = recall_score(p.label_ids, preds, average="macro")

    return {
        "accuracy": accuracy,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
    }


def tokenize_function_nsp(examples, tokenizer, max_length):
    return tokenizer(
        examples["sentenceA"],
        examples["sentenceB"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
