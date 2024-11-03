from sklearn.metrics import confusion_matrix
import pandas as pd

from utils import set_seed
from pipeline import TrainTestVal
from finbert_gpt import predict_finbertgpt, predict_nsp_batch
from pipeline import create_nsp_labels 

set_seed(42)

data = TrainTestVal(
    gpt_path="data/gpt4",
    phrasebank_path="data/sentences_50agree",
    test_size=0.2,
    is_synthetic_in_test=False,
    trace=True,
    from_hub=False,
    seed=42,
)


path = f"../saved_models/lambda-labs-nsp-200000"
test = create_nsp_labels(
                path="data/nsp/bloombery_test_long_nsp.csv", size=2500, seed=42
            )

result = predict_nsp_batch(model_name=path, test_df=test, is_nsp=True)
cmatrix = confusion_matrix(y_pred=result["predicted"], y_true=result["label"])
print(cmatrix)

test['predicted'] = result['predicted']
filter_row = test[test["label"] != test["predicted"]]
res = filter_row.iloc[0]
print((res['sentenceA'], res['sentenceB'], res['label']))
res = filter_row.iloc[1]
print((res['sentenceA'], res['sentenceB'], res['label']))
