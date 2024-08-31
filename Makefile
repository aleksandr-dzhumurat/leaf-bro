CURRENT_DIR = $(shell pwd)
NETWORK_NAME = service_network
 
include .env
export


prepare-dirs:
	mkdir -p ${CURRENT_DIR}/data || true & \
	mkdir -p ${CURRENT_DIR}/data/pipelines-data || true & \
	mkdir -p ${CURRENT_DIR}/data/api-data || true & \
	mkdir -p ${CURRENT_DIR}/data/pipelines-data/models || true & \
	mkdir -p ${CURRENT_DIR}/data/es-data || true & \
	mkdir -p ${CURRENT_DIR}/data/dagster-data || true

elastic:
	docker-compose up elasticsearch

run-jupyter:
	jupyter notebook jupyter_notebooks --ip 0.0.0.0 --port 8887 \
	--NotebookApp.token='' --NotebookApp.password='' --allow-root --no-browser

clear-data:
	rm -rf ${CURRENT_DIR}/data/pipelines-data/*

run-dagster:
	DAGSTER_HOME=${CURRENT_DIR}/data/dagster-data dagster dev

push-api:
	docker build -f services/api/Dockerfile -t adzhumurat/leaf_bro_api . && \
	docker push adzhumurat/leaf_bro_api:latest