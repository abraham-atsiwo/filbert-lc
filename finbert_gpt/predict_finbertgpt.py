from typing import Dict

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np 
from torch.utils.data import Dataset

from utils import timer_decorator


class NSPDataset(Dataset):
    def __init__(self, textsA, textsB, labels, tokenizer):
        self.textsA = textsA
        self.textsB = textsB
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.textsA)

    def __getitem__(self, idx):
        textA = self.textsA[idx]
        textB = self.textsB[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            textA, textB, truncation=True, padding="max_length", return_tensors="pt"
        )

        return {key: value.squeeze() for key, value in encoding.items()}, label


def create_dataloader(test_df, tokenizer, batch_size=16):
    test_textsA = test_df["sentenceA"].to_list()
    test_textsB = test_df["sentenceB"].to_list()
    labels = test_df["label"].to_list()

    dataset = NSPDataset(test_textsA, test_textsB, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader


@timer_decorator
def predict_nsp_batch(
    model_name: str,
    test_df: pd.DataFrame,
    device="mps",
    seed: int = None,
    batch_size=16,
    return_df: bool = False,
    is_nsp: bool = True,
):
    if seed:
        set_seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataloader = create_dataloader(test_df, tokenizer, batch_size)
    device = torch.device(
        device
        if torch.backends.mps.is_built() and device == "mps"
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    all_labels = []
    all_preds = []
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.clone().detach().to(device)
            # labels = torch.tensor(labels).to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    total_loss /= len(dataloader.dataset)
    if is_nsp:
        report = classification_report(
            all_labels, all_preds, target_names=["notNext", "isNext"], digits=4
        )
    else:
        report = classification_report(
            all_labels,
            all_preds,
            target_names=["negative", "neutral", "positive"],
            digits=4,
        )
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    if return_df:
        test_df["predicted_label"] = all_preds
        return test_df
    return {
        "report": report,
        "loss": total_loss,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "predicted": all_preds,
        "label": all_labels,
    }


def predict_finbertgpt(
    model_name: str,
    test_df: pd.DataFrame,
    device="mps",
    test_id2label: Dict[int, str] = None,
    seed: int = None,
):

    test_df["label"] = np.int64(test_df["label"])
    if seed:
        set_seed(seed)
    if test_id2label is None:
        test_id2label = {0: "negative", 1: "neutral", 2: "positive"}
        
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_textsA = test_df["sentence"].to_list()
    labels = test_df["label"].to_numpy()

    encoded_inputs = tokenizer(
        test_textsA, truncation=True, padding=True, return_tensors="pt"
    )
    # Move model and data to MPS device
    device = torch.device(
        device
        if torch.backends.mps.is_built() and device == "mps"
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)
    encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}
    labels = torch.tensor(labels).to(device)

    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**encoded_inputs)
        logits = outputs.logits.cpu()
        loss = criterion(outputs.logits, labels).item()

    # Get the predicted class
    predictions = torch.argmax(logits, dim=-1)

    all_labels = labels.cpu().numpy()
    all_preds = predictions.cpu().numpy()
    # convert id to labels
    id2label = model.config.id2label
    all_preds = [id2label[id].lower() for id in all_preds]
    all_labels = [test_id2label[id].lower() for id in all_labels]
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    precision_macro = precision_score(all_labels, all_preds, average="macro")

    # Return classification report and loss
    report = classification_report(y_true=all_labels, y_pred=all_preds)

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "loss": loss,
        "report": report,
        "label": all_labels,
        "predicted": all_preds,
        # 'sentence': test_textsA
    }
    return metrics
