### 1. `nsp-finetuned-bloombery` 

Pretrained model on Bloombery financial data using bi-directional representation from transformers (BERT) for next setence prediction (NSP) in finance. The BERT uncased version was finetuned for this task. 


### `Model description`

NSP-finetuned-bloombery is a transformers model trained on `200,000` sentence pairs in a self surpervised fashion. It was trained on raw text input only. It was trained for one main objective, next sentence prediction. 

- `Next Sentence Prediction:` The model was finetuned on `200,000` sentence pairs. The target variable is created by splitting a long sequence into two parts. A label of one is assigned by taking the actual next sentence as sentenceB and the previous sentence as sentenceA. 


### Intended uses & Limitations

This model can be further finetuned for financial fask because of the rich financial language learned during finetuning. 

### Model Weights

- Download Model Weights from HuggingFace: [Download Weight](https://huggingface.co/ab30atsiwo/nsp-finetuned-bloombery/tree/main)

```python
# Use a pipeline as a high-level helper
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
pipe = pipeline("text-classification", model="ab30atsiwo/nsp-finetuned-bloombery")

tokenizer = AutoTokenizer.from_pretrained("ab30atsiwo/nsp-finetuned-bloombery")
model = AutoModelForSequenceClassification.from_pretrained("ab30atsiwo/nsp-finetuned-bloombery")
```




### 2. Finbert-LC: Finetuned BERT model on Real and Synthetic Generated Data for Financial Sentiment Analysis

This project involves fine-tuning a BERT (Bidirectional Encoder Representations from Transformers) model on both real and synthetically generated data to enhance financial sentiment analysis. This approach leverages pre-trained language models and aims to improve sentiment detection accuracy specifically within the financial domain.


### Features

•	Pre-trained BERT fine-tuning tailored for financial sentiment analysis.

•	Synthetic data generation to augment real data, enhancing model performance.

•	Sentiment classification (e.g., positive, negative, neutral) specific to finance-related text.

•	Evaluation metrics to assess model accuracy and robustness in real-world financial sentiment analysis.


### Data

The dataset includes both real financial data and synthetic data generated to represent possible real-world scenarios. Synthetic data aims to address domain-specific challenges, such as limited training data in financial sentiment contexts.


### License

- This project is licensed under the MIT License. See the LICENSE file for details.

- This template should fit well for your project. Let me know if you’d like more customization!


### Model Weights

- Download Model Weights from HuggingFace: [Download Weight](https://huggingface.co/ab30atsiwo/finbert-gpt-allagree/tree/main)

```python

from transformers import (BertTokenizer, 
                          BertForSequenceClassification)
import torch

model_name = "ab30atsiwo/finbert-gpt-allagree"
# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  

# An example stock
texts = ["This stock is performing well!", "The market is crashing."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
print(outputs)
```
