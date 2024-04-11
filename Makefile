.PHONY: mlflow pipeline build

build:
	docker build -t mlflow-server -f images/mlflow.Dockerfile .
	docker build -t preprocessing -f images/preprocessing.Dockerfile .
	docker build -t training -f images/training.Dockerfile .
	docker build -t serving -f images/serving.Dockerfile .
	docker network create mlops-network

mlflow: build
	docker run -d -p 5000:5000 -v ./mlruns:/mlruns -v ./mlartifacts:/mlartifacts --network mlops-network \
 	--name mlflow mlflow-server

experiment:
	docker run --rm -v ./src/experiments:/src --network mlops-network --name experiments training python -m src.elastic

preprocessing:
	docker run --rm -v ./src/pipeline/preprocessing.py:/src/preprocessing.py -v ./data:/data --network mlops-network \
	--name prep preprocessing python -m src.preprocessing

training: preprocessing
	docker run --rm -v ./src/pipeline/training.py:/src/training.py  -v ./data:/data --network mlops-network \
	--name train training python -m src.training

pipeline: training

serve:
	docker run -d -p 8080:8080 -v ./src/api/app.py:/src/app.py --network mlops-network --name api serving

stop-mlflow:
	docker rm -f mlflow
	docker network rm mlops-network
