
# mlflow-spark-summit-2019 - best_run

Finds the best run of an experiment by searching for the min or max of a metric.

Ideally we would like to execute the search on the server side for scalability reasons.
Since the [search](https://www.mlflow.org/docs/latest/search-syntax.html) syntax does not support min/max, we have to perform the search logic on the client side.

Two implementations:
* Slow - Finds the best run by calling get_run for each run. Optimized for space as response payloads are small.
* Fast - Finds the best run by calling search once to get data for all an experiment's runs. Optimized for time but response payload will be large for experiments with many runs.

Sample run for [best_run.py](best_run.py):
```
python  best_metric.py --experiment_id 2 --metric rmse --ascending
```
```
slow best: ('3d57e49ba31843ac9ea3f4443ac4fbac', 0.7585747707504502)
fast best: ('3d57e49ba31843ac9ea3f4443ac4fbac', 0.7585747707504502)
```
