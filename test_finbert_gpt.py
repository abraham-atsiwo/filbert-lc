import os
import sys
from collections import Counter
from typing import NamedTuple

from datasets import load_dataset
import pandas as pd

from finbert_gpt import FinbertGPT, predict_nsp_batch, predict_finbertgpt
from pipeline import TrainTestVal

os.environ["TOKENIZERS_PARALLELISM"] = "false"


data = load_dataset("ab30atsiwo/sentences_50agree_gpt")

data["test"].to_pandas().to_csv("test_lambda_models.csv", index=False)

data = pd.read_csv("test_lambda_models.csv")


data = TrainTestVal(
    gpt_path="data/gpt4",
    phrasebank_path="data/sentences_50agree",
    test_size=0.2,
    is_synthetic_in_test=False,
    trace=True,
    from_hub=False,
    seed=42,
)

path = f"../saved_models/finbert_gpt_freeze_cloud/finbert_50agree_main_pretrained"
model = FinbertGPT(
    # model_name="bert-base-uncased",
    model_name="../saved_models/lambda-labs-nsp-200000",
    data=data,
    max_length=512,
    is_nsp=False,
    seed=42,
)

model.trainer_hg(
    batch_size=8,
    epochs=3,
    freeze_embedding=False,
    freeze_layer=False,
    freeze_output=False,
    layer_num=[],
    save_local=True,
    trace=True,
    save_path=path,
    metric_for_best_model="loss",
)

print(path)
res = model.predict_hg()
print(res)

if __name__ == "__main___":

    data = TrainTestVal(
        gpt_path="data/gpt4",
        phrasebank_path="data/sentences_50agree",
        test_size=0.2,
        is_synthetic_in_test=False,
        trace=True,
        from_hub=False,
        seed=42,
    )

    model = FinbertGPT(
        model_name="bert-base-uncased",
        data=data,
        max_length=512,
        is_nsp=False,
        seed=42,
    )

    class TrainVal(NamedTuple):
        train: pd.DataFrame
        val: pd.DataFrame
        test: pd.DataFrame

    freeze_type = ["embedding", "output", "layer"]
    max_len = 512
    batch = 8
    for (
        i,
        freeze,
    ) in enumerate(freeze_type):
        data = load_dataset("ab30atsiwo/sentences_50agree_gpt")
        print(data)
        data = TrainVal(
            train=data["train"].to_pandas(),
            val=data["validation"].to_pandas(),
            test=data["test"].to_pandas(),
        )

        model = FinbertGPT(
            model_name="bert-base-uncased",
            data=data,
            max_length=max_len,
            is_nsp=False,
            seed=42,
        )
        path = f"../saved_models/finbert_50agree/{freeze}"
        if i == 0:
            model.trainer_hg(
                batch_size=batch,
                epochs=3,
                freeze_embedding=True,
                freeze_layer=False,
                freeze_output=False,
                layer_num=[0],
                save_local=True,
                trace=True,
                save_path=path,
                metric_for_best_model="loss",
            )
        elif i == 1:
            model.trainer_hg(
                batch_size=batch,
                epochs=3,
                freeze_embedding=False,
                freeze_layer=False,
                freeze_output=True,
                layer_num=[0],
                save_local=True,
                trace=True,
                save_path=path,
                metric_for_best_model="loss",
            )
        else:
            for j in range(0, 12):
                path = f"../saved_models/finbert_50agree_layer/{j}"
                model = FinbertGPT(
                    model_name="bert-base-uncased",
                    data=data,
                    max_length=max_len,
                    is_nsp=False,
                    seed=42,
                )
                model.trainer_hg(
                    batch_size=batch,
                    epochs=3,
                    freeze_embedding=False,
                    freeze_layer=True,
                    freeze_output=False,
                    layer_num=[j],
                    save_local=True,
                    trace=True,
                    save_path=path,
                    metric_for_best_model="loss",
                )

                print(path)
                res = model.predict_hg()
                print(res)
        # print Result
        print(path)
        res = model.predict_hg()
        print(res)

