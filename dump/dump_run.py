
"""
Recursively dumps all information a run including params, metrics, tags and artifacts.
"""

from __future__ import print_function
from argparse import ArgumentParser
import mlflow
from dump_utils import *

print("MLflow Version:", mlflow.version.VERSION)
  
def get_runs(client, infos, artifact_max_level):
    for info in infos:
        run = client.get_run(info.run_uuid)
        dump_run(run)
        dump_artifacts(client, info.run_uuid,"",INDENT_INC,artifact_max_level)

def dump(run_id, artifact_max_level):
    print("Options:")
    print("  run_id:",run_id)
    print("  artifact_max_level:",artifact_max_level)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    dump_run(run)
    dump_artifacts(client, run_id,"",INDENT_INC, artifact_max_level)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_id", dest="run_id", help="Run ID", required=True)
    parser.add_argument("--artifact_max_level", dest="artifact_max_level", help="Number of artifact levels to recurse", required=False, default=1, type=int)
    args = parser.parse_args()
    dump(args.run_id, args.artifact_max_level)
