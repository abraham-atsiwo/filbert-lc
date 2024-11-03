from typing import List

from transformers import (
    Trainer,
    AutoConfig,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import classification_report

from pipeline.load_data import TrainTestVal
from utils import timer_decorator, set_seed
from .utils import (
    compute_metrics,
    tokenize_function,
    tokenize_function_nsp,
    # timer_decorator,
    # set_seed,
)


class FinbertGPT:
    def __init__(
        self,
        model_name: str = None,
        data: TrainTestVal = None,
        max_length: int = None,
        is_nsp: bool = False,
        seed: int = None,
    ) -> None:
        if seed:
            set_seed(seed)
        if not is_nsp:
            lbl2id = {"negative": 0, "neutral": 1, "positive": 2}
            id2label = {value: key for key, value in lbl2id.items()}
        else:
            lbl2id = {"isNext": 0, "notNext": 1}
            id2label = {value: key for key, value in lbl2id.items()}
        self.id2label = id2label
        config = AutoConfig.from_pretrained(model_name, num_labels=len(lbl2id))
        config.label2id = lbl2id
        config.id2label = id2label
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # datasets
        if isinstance(data.train, Dataset):
            train_dataset = data.train
            test_dataset = data.test
            val_dataset = data.val
        else:
            train_dataset = Dataset.from_pandas(data.train)
            val_dataset = Dataset.from_pandas(data.val)
            test_dataset = Dataset.from_pandas(data.test)
            self.test_label = test_dataset["label"]

        if is_nsp:
            tokenize_func = tokenize_function_nsp
        else:
            tokenize_func = tokenize_function
        self.train = train_dataset.map(
            tokenize_func,
            fn_kwargs={"tokenizer": self.tokenizer, "max_length": max_length},
            batched=True,
        )
        self.val = val_dataset.map(
            tokenize_func,
            fn_kwargs={"tokenizer": self.tokenizer, "max_length": max_length},
            batched=True,
        )
        self.test = test_dataset.map(
            tokenize_func,
            fn_kwargs={"tokenizer": self.tokenizer, "max_length": max_length},
            batched=True,
        )

        self.training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=2e-5,
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            # logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=50,
            save_steps=2000,
            # push_to_hub=True,
            # push_to_hub_model_id="",
            save_strategy="steps",
            log_level="error",
            save_total_limit=0,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            # push_to_hub_token="",
        )

        self.trainer = None

    def _get_layer_num(self, param_name):
        # This helper function extracts the layer number from the parameter name
        if "bert.encoder.layer" in param_name:
            parts = param_name.split(".")
            try:
                layer_index = parts.index("layer")
                return int(parts[layer_index + 1])
            except (ValueError, IndexError):
                return None
        return None

    @timer_decorator
    def trainer_hg(
        self,
        batch_size: int,
        epochs: int,
        freeze_layer: bool = True,
        freeze_embedding: bool = False,
        freeze_output=True,
        layer_num: List[int] = 0,
        save_path: str = None,
        trace: bool = True,
        save_local: str = False,
        metric_for_best_model: str = None,
    ) -> None:
        layer_num = [j - 1 for j in layer_num]
        if metric_for_best_model is None:
            metric_for_best_model == "accuracy"
        metric_for_best_model = metric_for_best_model.lower()
        if metric_for_best_model == "loss":
            self.training_args.metric_for_best_model = metric_for_best_model
            self.training_args.greater_is_better = False
        else:
            self.training_args.metric_for_best_model = metric_for_best_model
            self.training_args.greater_is_better = True
        self.training_args.per_device_eval_batch_size = batch_size
        self.training_args.per_device_train_batch_size = batch_size
        self.training_args.num_train_epochs = epochs
        if freeze_layer:
            for name, param in self.model.named_parameters():
                # Extract the layer number from the parameter name
                lay_num = self._get_layer_num(name)
                if lay_num in layer_num:
                    param.requires_grad = False
        # Print the names and status of parameters to
        if freeze_embedding:
            # Freeze the embedding layer
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False

        # Freeze the output layer (classifier layer)
        if freeze_output:
            for param in self.model.classifier.parameters():
                param.requires_grad = False
            for param in self.model.bert.pooler.parameters():
                param.requires_grad = False

        if trace:
            for name, param in self.model.named_parameters():
                print(f"{name}: {'Frozen' if not param.requires_grad else 'Trainable'}")

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train,
            eval_dataset=self.val,
            compute_metrics=compute_metrics,
        )
        self.trainer.train()
        # Evaluate the model
        self.trainer.evaluate()
        # Save the model locally
        if save_local:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

    @timer_decorator
    def predict_hg(self, test=None):
        predictions = self.trainer.predict(self.test)
        # Get the predicted class
        predicted_classes = np.argmax(predictions.predictions, axis=-1)
        predicted_classes = [self.id2label[j] for j in predicted_classes]
        test_label = [self.id2label[lbl] for lbl in self.test_label]
        return classification_report(test_label, predicted_classes)
