from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess, split_data, train_model, evaluate_model

def create_pipeline(**kwargs):
    return pipeline([
        node(preprocess, ["raw_data", "parameters"], "processed_data"),
        node(split_data, ["processed_data", "parameters"],
             ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]),
        node(train_model, ["X_train", "y_train", "parameters"], "trained_model"),
        node(evaluate_model, ["trained_model", "X_val", "y_val"], "metrics"),
    ])