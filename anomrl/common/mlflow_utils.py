import mlflow

def setup_experiment(exp_name: str):
    try:
        experiment = mlflow.get_experiment_by_name(exp_name)
        assert experiment is not None
        experiment_id = experiment.experiment_id
    except Exception:
        experiment_id = mlflow.create_experiment(name=exp_name)

    mlflow.set_experiment(experiment_name=exp_name)
    return experiment_id
