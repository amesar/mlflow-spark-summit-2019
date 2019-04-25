
"""
Recursively dumps all information about an experiment including all details of its runs and their params, metrics and artifacts.
Note that this can be expensive. Adjust your artifact_max_level.
"""

from __future__ import print_function
import sys
from argparse import ArgumentParser
import mlflow
from dump_utils import *

print("MLflow Version:", mlflow.version.VERSION)

def dump_experiment(exp):
    print("Experiment Details:")
    for k,v in exp.__dict__.items(): print("  {}: {}".format(k[1:],v))
  
def get_runs(client, infos, artifact_max_level):
    for info in infos:
        run = client.get_run(info.run_uuid)
        dump_run(run)
        dump_artifacts(client, info.run_uuid,"",INDENT_INC,artifact_max_level)

def dump(exp_id_or_name, artifact_max_level, show_runs):
    print("Options:")
    print("  exp_id_or_name:",exp_id_or_name)
    print("  artifact_max_level:",artifact_max_level)
    print("  show_runs:",show_runs)
    client = mlflow.tracking.MlflowClient()
    if exp_id_or_name.isdigit():
        exp_id = int(exp_id_or_name)
    else:
        print("experiment_name:",exp_id_or_name)
        exp_id = client.get_experiment_by_name(exp_id_or_name).experiment_id
    print("experiment_id:",exp_id)
    exp = client.get_experiment(exp_id)
    dump_experiment(exp)
    infos = client.list_run_infos(exp_id)
    print("  #runs:",len(infos))
    if not show_runs: 
        return
    get_runs(client,infos,artifact_max_level)
    print("#runs:",len(infos))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_id_or_name", dest="experiment_id", help="Experiment ID", required=True)
    parser.add_argument("--artifact_max_level", dest="artifact_max_level", help="Number of artifact levels to recurse", required=False, default=1, type=int)
    parser.add_argument("--show_runs", dest="show_runs", help="Show runs", required=False, default=False, action='store_true')
    args = parser.parse_args()
    dump(args.experiment_id, args.artifact_max_level,args.show_runs)
