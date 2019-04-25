
from __future__ import print_function
import sys
import mlflow
import mlflow.spark as mlflow_spark
from pyspark.sql import SparkSession

print("MLflow Version:", mlflow.version.VERSION)
print("Tracking URI:", mlflow.tracking.get_tracking_uri())

if __name__ == "__main__":
    run_id = sys.argv[1]
    print("run_id:",run_id)
    spark = SparkSession.builder.appName("Predict").getOrCreate()

    data_path = "../data/sample_libsvm_data.txt"
    print("data_path:",data_path)
    data = spark.read.format("libsvm").load(data_path)

    model = mlflow_spark.load_model("spark-model", run_id=run_id)
    predictions = model.transform(data)

    print("Prediction Dataframe")
    predictions.printSchema()

    print("Filtered Prediction Dataframe")
    df = predictions.select("prediction", "indexedLabel","probability").filter("prediction <> indexedLabel")
    df.printSchema()
    df.show(5,False)
