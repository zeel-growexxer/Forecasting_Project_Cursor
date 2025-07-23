import mlflow

def set_experiment(experiment_name):
    mlflow.set_experiment(experiment_name)

def start_run(run_name=None):
    return mlflow.start_run(run_name=run_name)

def log_params(params):
    mlflow.log_params(params)

def log_metrics(metrics):
    mlflow.log_metrics(metrics)

def log_model(model, artifact_path):
    mlflow.pytorch.log_model(model, artifact_path)
