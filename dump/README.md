# mlflow-spark-summit-2019 - dump

Dumps all experiment or run information recursively.

**Dump Run**

* [dump_run.py](dump_run.py) 
* [sample dump](run.txt)

```
python dump_run.py --run_id 2cbab69842e4412c99bfb5e15344bc42 --artifact_max_level 5 
```

**Dump Experiment**

* [dump_experiment.py](dump_experiment.py) 
* [sample dump](experiment.txt)

```
python dump_experiment.py --experiment_id 2 --showRuns --artifact_max_level 5
```
