
# Serve predictions with mlflow.sklearn.load_model()

from __future__ import print_function
import sys
import mlflow
import mlflow.sklearn
import util

if __name__ == "__main__":
    if len(sys.argv) < 1:
        println("ERROR: Expecting RUN_ID PREDICTION_FILE")
        sys.exit(1)
    print("MLflow Version:", mlflow.version.VERSION)
    run_id = sys.argv[1]
    data_path = sys.argv[2] if len(sys.argv) > 2 else "data/wine-quality-red.csv"
    print("data_path:",data_path)
    print("run_id:",run_id)

    model = mlflow.sklearn.load_model("model", run_id=run_id)
    print("model:",model)

    df = util.read_prediction_data(data_path)
    predictions = model.predict(df)
    print("predictions:",predictions)
