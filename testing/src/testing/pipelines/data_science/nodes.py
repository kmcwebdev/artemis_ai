from transformers import AutoTokenizer
import json
import os
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from torch.optim import AdamW
from transformers.optimization import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
import logging
from typing import Dict, Tuple
from datasets import Dataset
from datasets import load_from_disk
from typing import Dict
from kedro.io import AbstractDataset

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from kedro_datasets.partitions import PartitionedDataset


# def split_data(encoded_dir: str, target_col: str, parameters: Dict) -> Dict[str, pd.DataFrame]:
#     test_size = parameters["test_size"]
#     random_state = parameters["random_state"]
#     partitioned_data = {}

#     for value in unique_values:
#         input_filepath = os.path.join(encoded_dir, f"{value}.csv")
#         df = pd.read_csv(input_filepath)
#         train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
#         partitioned_data[f"{value}_train"] = train_df
#         partitioned_data[f"{value}_test"] = test_df

#     return partitioned_data

def split_department_data(df: pd.DataFrame, parameters: Dict):
    train_df, test_df = train_test_split(
        df, test_size=parameters["test_size"], random_state=parameters["random_state"])
    return train_df, test_df


def split_df_to_dataset(encoded_dir: PartitionedDataset, parameters: Dict) -> Dict[str, Dict[str, pd.DataFrame]]:
    test_size = parameters["test_size"]
    random_state = parameters["random_state"]
    partitioned_data = {
        "train": {},
        "test": {}
    }
    for tech_group, partition_load_func in encoded_dir.items():
        # Load the data for the current partition
        partition_data = partition_load_func()

        # Split the partitioned data into train and test sets
        train_df, test_df = train_test_split(
            partition_data, test_size=test_size, random_state=random_state)
        train_df, test_df = dataframe_to_dataset(train_df, test_df)
        # Store the train and test sets in partitioned_data dictionary
        partitioned_data["train"][tech_group] = train_df
        partitioned_data["test"][tech_group] = test_df

    return partitioned_data["train"], partitioned_data["test"]


def split_techgroup_data(encoded_dir: str, parameters: Dict) -> Dict[str, pd.DataFrame]:
    return split_data(encoded_dir, parameters=parameters)


def split_subcategory_data(encoded_dir: str, parameters: Dict) -> Dict[str, pd.DataFrame]:
    return split_data(encoded_dir, target_col="Sub-Category", parameters=parameters)


def split_category_data(encoded_dir: str, parameters: Dict) -> Dict[str, pd.DataFrame]:
    return split_data(encoded_dir, target_col="Category", parameters=parameters)


def dataframe_to_dataset(train, test):
    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)
    return train_dataset, test_dataset


def department_label_encoding(train_dataset) -> Tuple[Dict[str, int], Dict[int, str]]:
    output_dir = "data/04_feature/department_label_encoded_dir"
    os.makedirs(output_dir, exist_ok=True)
    labels = [col for col in train_dataset.column_names if col != 'Description']

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    return label2id, id2label


model_path = 'microsoft/deberta-v3-small'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)


def preprocess_function(train_dataset, test_dataset, department2id):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def process_dataset(dataset, department2id):
        texts = dataset["Description"]
        departments = [
            col for col in dataset.column_names if col != 'Description']
        labels_list = []

        for example in dataset:
            labels = [0. for _ in range(len(departments))]
            for department in departments:
                if example[department] == 1:
                    label_id = department2id[department]
                    labels[label_id] = 1.
            labels_list.append(labels)

        encoded_texts = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=512
        )

        encoded_dict = {key: val for key, val in encoded_texts.items()}
        encoded_dict['labels'] = labels_list

        return Dataset.from_dict(encoded_dict)

    tokenized_train_dataset = process_dataset(train_dataset, department2id)
    tokenized_test_dataset = process_dataset(test_dataset, department2id)

    return tokenized_train_dataset, tokenized_test_dataset


def train_model(tokenized_train_dataset, tokenized_test_dataset, label2id, id2label, model_path='microsoft/deberta-v3-small'):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification"
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define metrics computation
    def compute_metrics(eval_pred):
        labels = eval_pred.label_ids
        predictions = eval_pred.predictions.argmax(-1)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='micro')
        precision = precision_score(labels, predictions, average='micro')
        recall = recall_score(labels, predictions, average='micro')
        return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

    # Training arguments with reduced verbosity
    training_args = TrainingArguments(
        output_dir="model",
        learning_rate=2e-5,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir='logs',  # Reduced verbosity in logging
        logging_strategy='steps',  # Reduced verbosity in logging
    )

    # Initialize the optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )

    # Train and save the model
    trainer.train()
    trainer.save_model()

    return model


# def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
#     """Trains the linear regression model.

#     Args:
#         X_train: Training data of independent features.
#         y_train: Training data for price.

#     Returns:
#         Trained model.
#     """
#     regressor = LinearRegression()
#     regressor.fit(X_train, y_train)
#     return regressor


def evaluate_model(
    regressor: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    y_pred = regressor.predict(X_test)
    score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    me = max_error(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
    return {"r2_score": score, "mae": mae, "max_error": me}
