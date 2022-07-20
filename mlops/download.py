import wandb
import pandas as pd
import os
import joblib
from config import WandbSettings


def download_data(dataset: str):
    """download the data from wandb

    Returns:
        _type_: dataset or file coming from outer programa
    """
    # os.system(f"wandb login --relogin ")
    run = wandb.init(project="mnist_deep_classification")
    artifact = run.use_artifact(f"mnist_deep_classification/{dataset}:latest")
    return artifact.file()


def download_model():
    """download of the model from wandb

    Returns:
        _type_: dataset or file coming from outer program
    """
    # os.system(f"wandb login --relogin ")
    run = wandb.init(project="mnist_deep_classification")
    artifact = run.use_artifact(
        "diego25rm/mnist_deep_classification/model_export:v0", type="pipeline_artifact"
    )
    return joblib.load(artifact.file())
