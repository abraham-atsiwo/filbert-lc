### `nsp-finetuned-bloombery` 

Pretrained model on Bloombery financial data using bi-directional representation from transformers (BERT) for next setence prediction (NSP) in finance. The BERT uncased version was finetuned for this task. 


### `Model description`

NSP-finetuned-bloombery is a transformers model trained on `200,000` sentence pairs in a self surpervised fashion. It was trained on raw text input only. It was trained for one main objective, next sentence prediction. 

- `Next Sentence Prediction:` The model was finetuned on `200,000` sentence pairs. The target variable is created by splitting a long sequence into two parts. A label of one is assigned by taking the actual next sentence as sentenceB and the previous sentence as sentenceA. 


### Intended uses & Limitations

This model can be further finetuned for financial fask because of the rich financial language learned during finetuning. 