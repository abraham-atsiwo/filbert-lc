from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt


# df = pd.read_csv("phrasebank_concatentate_gpt4.csv")


def token_distribution(df) -> None:
    df["token_count"] = df["sentence"].apply(count_tokens)
    plt.figure(figsize=(10, 6))
    plt.hist(
        df["token_count"],
        bins=range(df["token_count"].min(), df["token_count"].max() + 2),
        edgecolor="black",
    )
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")
    plt.show()


def count_tokens(text):
    return len(text.split())
