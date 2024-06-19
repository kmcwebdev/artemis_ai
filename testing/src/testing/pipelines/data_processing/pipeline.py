from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    preprocess_descriptions,
    create_department_df,
    create_techgroup_df,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func= preprocess_descriptions,
                inputs = "original_ref",
                outputs = "preprocessed_ref",
                name="preprocess_ref_node"
            ),
            node(
                func=create_department_df,
                inputs="preprocessed_ref",
                outputs="department_encoded_df",
                name="create_department_df_node"
            ),
            node(
                func=create_techgroup_df,
                inputs="preprocessed_ref",
                outputs="techgroup_encoded_dir",
                name="create_techgroup_df_node"
            ),
           
        ]
    )
