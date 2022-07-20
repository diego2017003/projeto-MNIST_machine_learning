import wandb


def upload_data_test(filename: str):
    """upload data from tests

    Args:
        filename (str): name of the file with the dataset
    """
    api = wandb.Api()
    run = api.run("mnist_predict/mnist_deep_classification/<run_id>")
    run.upload_file(filename)
