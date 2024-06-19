from kedro.pipeline import Pipeline, node, pipeline
from kedro.io import MemoryDataset
from kedro.runner import SequentialRunner

from .nodes import (split_data, 
    dataframe_to_dataset,
    department_label_encoding, 
    preprocess_function,
    train_model,

    )

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["department_encoded_df","params:model_options"],
                outputs=["train_df","test_df"],
                name="split_data_node"
            ),
            node(
                func=dataframe_to_dataset,
                inputs=["train_df","test_df"],
                outputs=["train_dataset", "test_dataset"],
                name="dataframe_to_dataset_node"
            ),
            node( 
                func=department_label_encoding,
                inputs="train_dataset",
                outputs=["label2id", "id2label"],
                name="department_label_encoding_node"
            ),
            node(
                func=preprocess_function,
                inputs=["train_dataset", "test_dataset","label2id"],
                outputs= ["tokenized_train_dataset", "tokenized_test_dataset"],
                name="tokenization_node"
            ),
            node(
                func=train_model,
                inputs=["tokenized_train_dataset", "tokenized_test_dataset", "label2id", "id2label"],
                outputs="trained_model",
                name="train_model_node"
            )
            # node(
            #     func=data_split,
            #     inputs=["model_input_table@pandas", "params:model_options"],
            #     outputs=["X_train", "X_test", "y_train", "y_test"],
            #     name="split_data_node",
            # ),
            # node(
            #     func=train_model,
            #     inputs=["X_train", "y_train"],
            #     outputs="regressor",
            #     name="train_model_node",
            # ),
            # node(
            #     func=evaluate_model,
            #     inputs=["regressor", "X_test", "y_test"],
            #     outputs="metrics",
            #     name="evaluate_model_node",
            # ),
        ]
    )
