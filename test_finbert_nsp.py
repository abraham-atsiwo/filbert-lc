from typing import NamedTuple
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 

from finbert_gpt import predict_nsp_batch, FinbertGPT
from pipeline import create_nsp_labels


class TrainTest(NamedTuple):
    train: pd.DataFrame = None
    val: pd.DataFrame = None
    test: pd.DataFrame = None


os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

df_nsp = create_nsp_labels(path="data/nsp/bloombery_nsp.parquet", size=100000)


train_data, val_data = train_test_split(df_nsp, test_size=0.2, random_state=42)


data = TrainTest(train=train_data, val=val_data, test=val_data)

model = FinbertGPT(
    model_name="bert-base-uncased", data=data, max_length=256, is_nsp=True, seed=42
)


model.trainer_hg(
    batch_size=16,
    epochs=3,
    freeze_embedding=False,
    freeze_layer=False,
    freeze_output=False,
    layer_num=[0],
    save_local=True,
    trace=False,
    save_path="../saved_models/nsp-bloombery",
    metric_for_best_model="loss",
)


model_name = "../saved_models/nsp-bloombery"
data = pd.read_csv("data/nsp/nsp_test_data.csv")
test_df = data 

# Perform NSP predictions with DataLoader
report, loss = predict_nsp_batch(model_name, test_df, seed=42)
print("Classification Report:\n", report)
print("Loss:", loss)