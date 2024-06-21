from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    preprocess_descriptions,
    create_department_df,
    create_techgroup_df,
    create_category_df,
    create_subcategory_df,
    merge_datasets
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
                func= merge_datasets,
                inputs = ["original_ref","cate_ref"],
                outputs = "merged_df",
                name = "merge_data_node"
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
            node(
                func=create_subcategory_df,
                inputs="preprocessed_ref",
                outputs="subcategory_encoded_dir",
                name="create_subcategory_df_node"
            ),
            node(
                func=preprocess_descriptions,
                inputs="merged_df",
                outputs="cate_preprocessed_ref",
                name="preprocess_cate_ref_node"
            ),
            node(
                func=create_category_df,
                inputs="cate_preprocessed_ref",
                outputs="category_encoded_dir",
                name="create_category_df_node"
            ),

        ]
    )
