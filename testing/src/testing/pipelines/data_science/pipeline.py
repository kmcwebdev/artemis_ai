from kedro.pipeline import Pipeline, node, pipeline
# from kedro.io import MemoryDataset
# from kedro.runner import SequentialRunner

from .nodes import ( 
    dataframe_to_dataset,
    department_label_encoding, 
    preprocess_function,
    train_model,
    split_department_data,
    split_df_to_dataset
    # split_subcategory_data
    )

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_department_data,
                inputs=["department_encoded_df","params:model_options"],
                outputs=["train_department_df","test_department_df"],
                name="split_department_node"
            ),
            node(
                func=dataframe_to_dataset,
                inputs=["train_department_df", "test_department_df"],
                outputs=["train_department_dataset",
                         "test_department_dataset"],
                name="dataframe_to_dataset_node"
            ),
            node(
                func=department_label_encoding,
                inputs="train_department_dataset",
                outputs=["department2id", "id2department"],
                name="department_label_encoding_node"
            ),
            node(
                func=preprocess_function,
                inputs=["train_department_dataset",
                        "test_department_dataset", "department2id"],
                outputs=["tokenized_train_department_dataset", "tokenized_test_department_dataset"],
                name="department_tokenization_node"
            ),
            node(
                func=train_model,
                inputs=["tokenized_train_department_dataset",
                        "tokenized_test_department_dataset", "department2id", "id2department"],
                outputs="trained_model",
                name="train_model_node"
            ),
            node(
                func=split_df_to_dataset,
                inputs=["techgroup_encoded_dir", "params:model_options"],
                outputs=["train_techgroup_dir","test_techgroup_dir"],
                name="split_techgroup_and_transform_to_dataset_node"
            ),
            node(
                func=split_df_to_dataset,
                inputs=["category_encoded_dir", "params:model_options"],
                outputs=["train_category_dir", "test_category_dir"],
                name="split_category_and_transform_to_dataset_node"
            ),
            node(
                func=split_df_to_dataset,
                inputs=["subcategory_encoded_dir","params:model_options"],
                outputs=["train_subcategory_dir", "test_subcategory_dir"],
                name="split_subcategory_and_transform_to_dataset_node"
            ),
            
            # node(
            #     func=split_data,
            #     inputs=["techgroup_encoded_dir","params:model_options"],
            #     outputs=["train", "test"]
            # )
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
