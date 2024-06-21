import logging
from typing import Dict, Tuple
from datasets import Dataset
from datasets import load_from_disk

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame, parameters:Dict):
    train_df, test_df = train_test_split(df, test_size=parameters["test_size"], random_state=parameters["random_state"])
    return train_df, test_df

# def split_subcategory_data(encoded_dir,df, parameters:Dict):
#     unique_departments = df['Department'].unique()
#     output_dir = "data/03_primary/subcategory_train_test_df_dir"
#     os.makedirs(output_dir, exist_ok=True)
#     for department in unique_departments:
#     # Load the encoded DataFrame for the specified department and subcategory
#         input_filepath = os.path.join(encoded_dir, f"{department}_encoded.csv")
#         df = pd.read_csv(input_filepath)
#         df_subcategory = df[df['Sub-Category'] == subcategory]
#         train_df, test_df = train_test_split(df_subcategory, test_size=parameters["test_size"], random_state=parameters["random_state"])
#         output_filepath = os.path.join(output_dir, f"{department}_sub.csv")
#     return 


from typing import Dict

def split_subcategory_data(encoded_dir: str, parameters: Dict) -> Dict[str, pd.DataFrame]:
    # Load parameters
    test_size = parameters["test_size"]
    random_state = parameters["random_state"]
    
    # Create a dictionary to store the partitioned data
    partitioned_data = {}

    # Iterate through each department's data in the encoded_dir
    for department in parameters["departments"]:
        input_filepath = os.path.join(encoded_dir, f"{department}.csv")
        df = pd.read_csv(input_filepath)
        
        # Split the data
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        
        # Store the train and test data in the partitioned_data dictionary
        partitioned_data[f"{department}_train"] = train_df
        partitioned_data[f"{department}_test"] = test_df
    
    return partitioned_data


def dataframe_to_dataset(train,test):
    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)
    return train_dataset, test_dataset

import os
import json
def department_label_encoding(train_dataset) -> Tuple[Dict[str, int], Dict[int, str]]:
    output_dir = "data/04_feature/department_label_encoded_dir"
    os.makedirs(output_dir, exist_ok=True)
    labels = [col for col in train_dataset.column_names if col != 'Description']

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    return label2id, id2label

from transformers import AutoTokenizer

model_path = 'microsoft/deberta-v3-small'
tokenizer = AutoTokenizer.from_pretrained(model_path)

def preprocess_function(train_dataset, test_dataset, department2id):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    def process_dataset(dataset, department2id):
        texts = dataset["Description"]
        departments = [col for col in dataset.column_names if col != 'Description']
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

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from torch.optim import AdamW

def train_model(tokenized_train_dataset, tokenized_test_dataset, label2id, id2label, model_path='microsoft/deberta-v3-small'):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=len(label2id),
        id2label=id2label, 
        label2id=label2id,
        problem_type="multi_label_classification"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        labels = eval_pred.label_ids
        predictions = eval_pred.predictions.argmax(-1)
        
        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='micro')
        precision = precision_score(labels, predictions, average='micro')
        recall = recall_score(labels, predictions, average='micro')

        return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}
    
    training_args = TrainingArguments(
        output_dir="model",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    # Initialize the optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

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
    
    trainer.train()
    trainer.save()
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
