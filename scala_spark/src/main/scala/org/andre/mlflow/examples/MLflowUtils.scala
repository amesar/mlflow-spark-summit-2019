package org.andre.mlflow.examples

import scala.collection.JavaConversions._
import org.mlflow.tracking.MlflowClient
import org.mlflow.tracking.creds.BasicMlflowHostCreds

object MLflowUtils {

  def getOrCreateExperimentId(client: MlflowClient, experimentName: String) : String = {
    val expOpt = client.listExperiments() find (_.getName == experimentName)
    expOpt match {
      case Some(exp) => exp.getExperimentId
      case None => client.createExperiment(experimentName)
    }
  }

  def createMlflowClient(args: Array[String]) = {
    println("args: "+args)
    if (args.length == 0) {
        val env = System.getenv("MLFLOW_TRACKING_URI")
        println(s"MLFLOW_TRACKING_URI: $env")
        new MlflowClient()
    } else {
      val trackingUri = args(0)
      println(s"Tracking URI: $trackingUri")
      if (args.length > 1) {
        new MlflowClient(new BasicMlflowHostCreds(trackingUri,args(1)))
      } else {
        new MlflowClient(trackingUri)
      }
    }
  }

  def createMlflowClient(trackingUri: String, token: String) = {
    if (trackingUri == null) {
        val env = System.getenv("MLFLOW_TRACKING_URI")
        println(s"MLFLOW_TRACKING_URI: $env")
        new MlflowClient()
    } else {
      if (token != null) {
        new MlflowClient(new BasicMlflowHostCreds(trackingUri, token))
      } else {
        new MlflowClient(trackingUri)
      }
    }
  }
}
