package org.andre.mlflow.examples.decisiontree

import com.beust.jcommander.{JCommander, Parameter}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.Transformer
import org.mlflow.tracking.MlflowClient
import org.mlflow.tracking.creds.BasicMlflowHostCreds
import org.andre.mlflow.examples.{MLflowUtils,MLeapUtils}

object PredictDecisionTree {

  def main(args: Array[String]) {
    new JCommander(opts, args.toArray: _*)
    println("Options:")
    println(s"  dataPath: ${opts.dataPath}")
    println(s"  tracking URI: ${opts.trackingUri}")
    println(s"  token: ${opts.token}")
    println(s"  runId: ${opts.runId}")

    val mlflowClient = MLflowUtils.createMlflowClient(opts.trackingUri, opts.token)
    val spark = SparkSession.builder.appName("Predict").getOrCreate()
    val data = spark.read.format("libsvm").load(opts.dataPath)

    val runInfo = mlflowClient.getRun(opts.runId).getInfo
    val uri = runInfo.getArtifactUri
    predictSparkML(uri, data)
    predictMLeap(uri, data)
  }

  def predictSparkML(uri: String, data: DataFrame) {
    println("==== Spark ML")
    val modelPath = s"${uri}/spark-model"
    val model = PipelineModel.load(modelPath)
    showPredictions(model, data)
  }

  def predictMLeap(uri: String, data: DataFrame) {
    println("==== MLeap")
    val modelPath = s"file:${uri}/mleap-model/mleap/model"
    val model = MLeapUtils.readModel(modelPath)
    showPredictions(model, data)
  } 

  def showPredictions(model: Transformer, data: DataFrame) {
    val predictions = model.transform(data)
    val df = predictions.select("prediction", "label", "features")
    df.show(10)
  }

  object opts {
    @Parameter(names = Array("--dataPath" ), description = "Data path", required=true)
    var dataPath: String = null

    @Parameter(names = Array("--trackingUri" ), description = "Tracking Server URI", required=false)
    var trackingUri: String = null

    @Parameter(names = Array("--token" ), description = "REST API token", required=false)
    var token: String = null

    @Parameter(names = Array("--runId" ), description = "runId", required=true)
    var runId: String = null
  }
}
