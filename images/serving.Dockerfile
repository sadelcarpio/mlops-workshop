FROM python:3.10

RUN pip install fastapi uvicorn mlflow
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]