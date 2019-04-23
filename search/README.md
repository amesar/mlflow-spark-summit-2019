
# mlflow-spark-summit-2019 - search

Synopsis:
* Shows how to search for runs. 
* For details see https://mlflow.org/docs/latest/search-syntax.html.

Summary of [search.py](search.py):
* Creates an experiment `search_example` with five runs with rmse values: 0.76, 0.71, 0.77, 0.69, 0.69.
* Searches for runs: `metrics.rmse >= 0.76`


Run
```
python search.py
```

```
MLflow Version: 0.9.1
Tracking URI: http://localhost:5000
experiment_name: search_example
experiment_id: 6
Adding 5 runs:
  metric: 0.76  run_uuid: cc2debe52ff14c6b9e87cbbe27bedc5b
  metric: 0.71  run_uuid: bc762d998463434e95e8ebdfa50019c0
  metric: 0.77  run_uuid: a9205f64570041b49d362b298195fcb6
  metric: 0.69  run_uuid: fe3aeb9865e7428495b4dca3b6745177
  metric: 0.69  run_uuid: 72d1281840334f3a934797ca196951a1
Query: metrics.rmse >= 0.76
Found 2 matching runs:
  run_uuid: a9205f64570041b49d362b298195fcb6  metrics: [<Metric: key='rmse', timestamp=1556059082, value=0.77>]
  run_uuid: cc2debe52ff14c6b9e87cbbe27bedc5b  metrics: [<Metric: key='rmse', timestamp=1556059082, value=0.76>]
```
