# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

from __future__ import print_function
import os
import sys
import platform

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, enet_path

import mlflow
import mlflow.sklearn
from wine_quality import plot_utils

print("MLflow Version:", mlflow.version.VERSION)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

class Trainer(object):
    def __init__(self, experiment_name, data_path, run_origin="none"):
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.run_origin = run_origin
        np.random.seed(40)

        print("experiment_name:",self.experiment_name)
        print("run_origin:",run_origin)

        # Read the wine-quality csv file 
        print("data_path:",data_path)
        data = pd.read_csv(data_path)
    
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)
    
        # The predicted column is "quality" which is a scalar from [3, 9]
        self.train_x = train.drop(["quality"], axis=1)
        self.test_x = test.drop(["quality"], axis=1)
        self.train_y = train[["quality"]]
        self.test_y = test[["quality"]]
        self.current_file = os.path.basename(__file__)

        self.X = data.drop(["quality"], axis=1).values
        self.y = data[["quality"]].values.ravel()

        # If using 'mlflow run' must use --experiment-id to set experiment since set_experiment() does not work
        if self.experiment_name != "none":
            mlflow.set_experiment(experiment_name)
            client = mlflow.tracking.MlflowClient()
            experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
            print("experiment_id:",experiment_id)

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def train(self, alpha, l1_ratio):
        with mlflow.start_run(source_name=self.current_file) as run:
            run_id = run.info.run_uuid
            print("run_id:",run_id)
            experiment_id = run.info.experiment_id
            print("  experiment_id:",experiment_id)
            clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            clf.fit(self.train_x, self.train_y)
    
            predicted_qualities = clf.predict(self.test_x)
            (rmse, mae, r2) = self.eval_metrics(self.test_y, predicted_qualities)
    
            #print("Parameters:(alpha={}, l1_ratio={}):".format(alpha, l1_ratio))
            print("  Parameters:")
            print("    alpha:",alpha)
            print("    l1_ratio:",l1_ratio)
            print("  Metrics:")
            print("    RMSE:",rmse)
            print("    MAE:",mae)
            print("    R2:",r2)
    
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
    
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            
            mlflow.set_tag("data_path", self.data_path)
            mlflow.set_tag("exp_id", experiment_id)
            mlflow.set_tag("exp_name", self.experiment_name)
            mlflow.set_tag("run_origin", self.run_origin)
            mlflow.set_tag("platform", platform.system())
    
            mlflow.sklearn.log_model(clf, "model")
    
            eps = 5e-3  # the smaller it is the longer is the path
            alphas_enet, coefs_enet, _ = enet_path(self.X, self.y, eps=eps, l1_ratio=l1_ratio, fit_intercept=False)
            plot_file = "wine_ElasticNet-paths.png"
            plot_utils.plot_enet_descent_path(self.X, self.y, l1_ratio, alphas_enet, coefs_enet, plot_file)
            mlflow.log_artifact(plot_file)
    
        return (experiment_id,run_id)
