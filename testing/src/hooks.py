import mlflow

from kedro_mlflow.hooks import MLflowLoggerHook


class CustomMLflowLoggerHook(MLflowLoggerHook):
    def after_catalog_created(self):
        mlflow.set_experiment(self.experiment_name)

    def after_node_run(self, node, catalog, inputs, outputs):
        mlflow.log_params(node.params)
        mlflow.log_metrics(node.metrics)
        mlflow.log_artifacts(outputs["predictions"],
                             artifact_path="predictions")
