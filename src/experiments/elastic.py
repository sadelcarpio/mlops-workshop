from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import pandas as pd
import numpy as np

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
try:
    mlflow.create_experiment("elasticnet-model")
except mlflow.exceptions.RestException:
    mlflow.set_experiment("elasticnet-model")

mlflow_experiment_id = mlflow.get_experiment_by_name("elasticnet-model").experiment_id

data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

alphas = [0.01, 0.03, 0.05]
l1_ratio = 0.5

train, test = train_test_split(data)
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train["quality"]
test_y = test[["quality"]]

for alpha in alphas:
    with mlflow.start_run(experiment_id=mlflow_experiment_id, run_name=f"run-alpha={alpha}"):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        predicted_qualities = lr.predict(test_x)
        rmse = np.sqrt(mean_squared_error(test_y, predicted_qualities))
        mae = mean_absolute_error(test_y, predicted_qualities)
        r2 = r2_score(test_y, predicted_qualities)

        print("Elasticnet Model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(lr, "model")

df = mlflow.search_runs(mlflow_experiment_id)
best_run_id = df.loc[df["metrics.rmse"].idxmin()]["run_id"]
mlflow.register_model(
    f"runs:/{best_run_id}/model", "best-elasticnet-model"
)
