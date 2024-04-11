from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlflow.set_tracking_uri("http://mlflow:5000")
current_timestamp = datetime.now()
try:
    mlflow.create_experiment("production-run")
except mlflow.exceptions.RestException:
    mlflow.set_experiment("production-run")

experiment_id = mlflow.get_experiment_by_name(f"production-run").experiment_id

with mlflow.start_run(experiment_id=experiment_id, run_name=f"{datetime.now().strftime('%Y-%m-%d')}") as run:
    X = pd.read_csv("../data/X.csv")
    y = pd.read_csv("../data/y.csv")
    model = mlflow.sklearn.load_model("models:/best-elasticnet-model/latest")
    model.fit(X, y)
    predicted_qualities = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predicted_qualities))
    mae = mean_absolute_error(y, predicted_qualities)
    r2 = r2_score(y, predicted_qualities)

    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    run_id = run.info.run_id
    mlflow.register_model(f"runs:/{run_id}/model", "production-model")
