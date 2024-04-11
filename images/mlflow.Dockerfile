FROM ghcr.io/mlflow/mlflow

RUN mlflow artifacts

CMD ["mlflow", "ui", "--host", "0.0.0.0"]