import pandas as pd
from datasets import load_dataset
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline import TestDFTokenCategories
from utils import generate_report
chunks = []

# Read the CSV file in chunks
for chunk in pd.read_csv(
    "data/twitter_test_df/twitter_btc_small.csv",
    chunksize=100,
    usecols=["hard_cleaned_text", "light_cleaned_text", "label"],
):
    # Perform operations on each chunk
    chunks.append(chunk)

# Combine chunks into a single DataFrame if needed
df = pd.concat(chunks, ignore_index=True)
print(len(df))

twitter_df = load_dataset(
    "zeroshot/twitter-financial-news-sentiment", trust_remote_code=True
)["validation"].to_pandas()[:100]
twitter_df.columns = ["sentence", "label"]
# print(twitter_df)
twitter_test_id2label = {0: "negative", 1: "positive", 2: "neutral"}

df = TestDFTokenCategories(
    directory="data/concat_data_test_history",
    source="concat_test",
    size_small=100,
    size_medium=300,
    size_large=400,
    same_len_per_class_small=True,
    same_len_per_class_medium=False,
    same_len_per_class_large=False,
    seed=42,
)

test = df.test_small
path = "../saved_models/lstm_phrasebank50agree/glove"
path = "../saved_models/finbert_gpt_freeze_cloud/finbert_50agree_main"
paths = "ProsusAI/finbert"
twitter_path = "NaturalStupidlty/FinBERT-Twitter-BTC"
paths = "yiyanghkust/finbert-tone"
# path = '../saved_models/finbert_gpt_freeze_cloud/finbert_50agree_layer_1'


if __name__ == "__main___":
    result = generate_report(
        model_name=path,
        test_df=twitter_df,
        device=None,
        batch_size=8,
        is_lstm_type=False,
        id2label=twitter_test_id2label,
    )
    print(result["report"])
