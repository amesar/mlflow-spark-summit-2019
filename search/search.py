from __future__ import print_function
import mlflow

print("MLflow Version:", mlflow.version.VERSION)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

experiment_name = "search_example"
print("experiment_name:",experiment_name)
mlflow.set_experiment(experiment_name)

client = mlflow.tracking.MlflowClient()
experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
print("experiment_id:",experiment_id)

def create_run(metric):
    with mlflow.start_run() as run:
        print("  metric:",metric," run_uuid:",run.info.run_uuid)
        mlflow.log_metric("rmse", metric)

def create_runs():
    metrics = [0.76, 0.71, 0.77, 0.69, 0.69 ]
    print("Adding {} runs:".format(len(metrics)))
    for m in metrics:
        create_run(m)
    run_infos = client.list_run_infos(experiment_id)

def delete_runs(experiment_id):
    run_infos = client.list_run_infos(experiment_id)
    for info in run_infos:
        client.delete_run(info.run_uuid)
    run_infos = client.list_run_infos(experiment_id)

def search(exp_ids, query):
    print("Query:",query)
    runs = client.search_runs(exp_ids,query)
    print("Found {} matching runs:".format(len(runs)))
    for run in runs:
        print("  run_uuid:",run.info.run_uuid," metrics:",run.data.metrics)

if __name__ == "__main__":
    delete_runs(experiment_id)
    create_runs()
    query = "metrics.rmse >= 0.76"
    search([experiment_id], query)

