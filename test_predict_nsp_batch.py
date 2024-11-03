from __future__ import division 

import pandas as pd

from finbert_gpt import predict_nsp_batch
from pipeline import create_nsp_labels


model_name = "ab30atsiwo/nsp-finetuned-bloombery"
model_name = "bert-base-uncased"
model_name = "../saved_models/lambda-labs-nsp-200000"
data = pd.read_csv("data/nsp/nsp_test_data.csv")
test_df = data



if __name__ == '__main__':

    df_res = pd.DataFrame(columns=["size","model_name","loss", "accuracy", "f1_macro"])
    # for s in [25, 50, 75, 100, 250, 500, 1000, 1500, 2000, 2500]:
    for s in [2500]:
        test_df_long = create_nsp_labels(
                path="data/nsp/bloombery_test_long_nsp.csv", size=s, seed=42
            )
        for model_name in [
            "bert-base-uncased",
            "bert-large-uncased",
            "../saved_models/lambda-labs-nsp-200000",
        ]:

            test_df = test_df_long
            print(model_name)
            result = predict_nsp_batch(
                model_name, test_df, seed=42, device="mps", batch_size=64
            )
            print(result["report"])
            del result["report"]
            result['size'] = s*2
            result['model_name'] = model_name
            df_res = pd.concat((df_res, pd.DataFrame([result])), ignore_index=True)
            # df_res.to_csv(path_or_buf='results_nsp/metrics_.csv')
            print(df_res)
