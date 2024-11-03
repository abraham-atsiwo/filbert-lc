import pandas as pd

from pipeline import TrainTestVal
from utils import count_bert_parameters, generate_report


data = TrainTestVal(
    gpt_path="data/gpt4",
    phrasebank_path="data/sentences_50agree",
    test_size=0.2,
    is_synthetic_in_test=False,
    trace=True,
    from_hub=False,
    seed=42,
)

test = data.test


def concat_dataframes(df1, df2):
    if df1.empty:
        return df2.dropna(
            axis=1, how="all"
        )  # Return df2 with all-NA columns dropped if df1 is empty
    else:
        # Drop all-NA columns before concatenation
        df1_cleaned = df1.dropna(axis=1, how="all")
        df2_cleaned = df2.dropna(axis=1, how="all")

        # Concatenate cleaned DataFrames
        return pd.concat([df1_cleaned, df2_cleaned], axis=0)


models = [
    "../saved_models/lstm_phrasebankallagree/bert",
    "../saved_models/lstm_phrasebankallagree/glove",
    "../saved_models/lstm_phrasebankallagree/fasttext",
    "../saved_models/finbert_allagree/gpt",
    "../saved_models/finbert_gpt_freeze_cloud/finbert_50agree_main",
]


def format_millions(x):
    if x >= 1e6:
        return f"{x / 1e6:.0f}M"
    return f"{x:.2f}"


models = (
    [f"../saved_models/finbert_gpt_freeze_cloud/embedding_layer"]
    + ["../saved_models/finbert_gpt_freeze_cloud/finbert_50agree_main_pretrained"]
)
if __name__ == "__main__":
    columns = ["model_name", "num_trainable" "loss", "accuracy", "f1_macro"]
    total_bert_pars = count_bert_parameters(layer_num=[j for j in range(12)])
    metrics_df = pd.DataFrame(columns=columns)
    for mod in models:
        path = mod.split("/")[-1]
        if path in ["glove", "fasttext", "bert", "elmo"]:
            is_lstm_path = True
        else:
            is_lstm_path = False
        result = generate_report(
            model_name=mod,
            test_df=test,
            device=None,
            batch_size=8,
            is_lstm_type=is_lstm_path,
        )

        loss = result["loss"]
        accuracy = result["accuracy"]
        f1_macro = result["f1_macro"]
        # calculate parameters
        layer_num = path.split("_")[-1]
        if layer_num == "layer":
            trainable = total_bert_pars["total"] - total_bert_pars["embedding"]
            model_name = "Embedding Layer"
        elif layer_num in [str(j) for j in range(12)]:
            tmp = count_bert_parameters(
                layer_num=[j for j in range(int(layer_num) + 1, 12)]
            )
            trainable = tmp["layer"] + tmp["output"]
            model_name = f"Layer {int(layer_num) + 1}"
        else:
            trainable = 0
            model_name = path
        res_df = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "num_trainable": format_millions(trainable),
                    "loss": loss,
                    "accuracy": accuracy,
                    "f1_macro": f1_macro,
                }
            ]
        )
        metrics_df = concat_dataframes(metrics_df, res_df)
        metrics_df = metrics_df.round(decimals=2)
        metrics_df.to_csv("model_results/finbert_lstm.csv")
        print(metrics_df)
    print(metrics_df.to_latex(float_format="%.2f", index=False))
