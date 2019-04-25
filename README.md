# mlflow-spark-summit-2019

MLflow code for Spark Summit 2019.

Session: [Managing the Complete Machine Learning Lifecycle with MLflow](https://databricks.com/sparkaisummit/north-america/sessions-single-2019?id=183).

## Setup
```
pip install mlflow==0.9.1
pip install matplotlib
pip install pyarrow
```

## MLflow Server
```
virtualenv mlflow_server
source mlflow_server/bin/activate
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri $PWD/mlruns --default-artifact-root $PWD/mlruns
```

## Examples
Before running an experiment:
```
export MLFLOW_TRACKING_URI=http://localhost:5000
```

* [hello_world](hello_world) - Hello World
* [sklearn](sklearn) - Scikit learn model
* [pyspark](pyspark) - PySpark model
* [scala_spark](scala_spark) - Scala Spark ML model using the Java client
* [search](search) - Shows new [MLflow 0.9.1 Search](https://mlflow.org/docs/latest/search-syntax.html) feature
* [dump](dump) - Shows usage of some [mlflow.tracking](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html) package methods
* [best_run](best_run) - Finds the best model run
