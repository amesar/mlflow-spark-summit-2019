package org.andre.mlflow.examples.hello

import java.io.{File,PrintWriter}
import org.mlflow.tracking.MlflowClient
import org.mlflow.tracking.creds.BasicMlflowHostCreds
import org.mlflow.api.proto.Service.RunStatus
import scala.collection.JavaConversions._
import org.andre.mlflow.examples.MLflowUtils

object HelloWorld {
  def main(args: Array[String]) {

    // Create MLflow client
    val mlflowClient = MLflowUtils.createMlflowClient(args)

    // Create or get existing experiment
    val expName = "scala_HelloWorld"
    val expId = MLflowUtils.getOrCreateExperimentId(mlflowClient, expName)
    println("Experiment name: "+expName)
    println("Experiment ID: "+expId)

    // Create run
    val sourceName = (getClass().getSimpleName()+".scala").replace("$","")
    val runInfo = mlflowClient.createRun(expId, sourceName)
    val runId = runInfo.getRunUuid()
    println("Run ID: "+runId)

    // Log params and metrics
    mlflowClient.logParam(runId, "p1","hi")
    mlflowClient.logMetric(runId, "m1",0.123)

    // Log file artifact
    new PrintWriter("info.txt") { write("File artifact: "+new java.util.Date()) ; close }
    mlflowClient.logArtifact(runId, new File("info.txt"))

    // Log directory artifact
    val dir = new File("tmp")
    dir.mkdir
    new PrintWriter(new File(dir, "model.txt")) { write("Directory artifact: "+new java.util.Date()) ; close }
    mlflowClient.logArtifacts(runId, dir, "model")

    // Close run
    mlflowClient.setTerminated(runId, RunStatus.FINISHED, System.currentTimeMillis())
  }
}
