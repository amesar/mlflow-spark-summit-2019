package org.andre.mlflow.examples

import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._
import resource.managed
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.DataFrame

/*
  MLeap URI formats:
    file:/tmp/mleap_scala_model_export/my-model
    jar:file:/tmp/mleap_scala_model_export/my-model.zip
*/
object MLeapUtils {

  def saveModel(model: PipelineModel, df: DataFrame, bundlePath: String) {
    val context = SparkBundleContext().withDataset(df)
    (for(modelFile <- managed(BundleFile(bundlePath))) yield {
      model.writeBundle.save(modelFile)(context)
    }).tried.get
  }

  def readModel(bundlePath: String) = {
    val obundle = (for(bundle <- managed(BundleFile(bundlePath))) yield {
      bundle.loadSparkBundle().get
    }).opt
    obundle match {
      case Some(b) => b.root
      case None => throw new Exception(s"Cannot find bundle: $bundlePath")
    }
  }
}
