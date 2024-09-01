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

build-network:
	docker network create backtier -d bridge || true

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

run-api-app:
	docker run -d --rm --network backtier --env-file ${CURRENT_DIR}/.env \
	-v ${CURRENT_DIR}/services/api/src:/srv/src \
	-v ${CURRENT_DIR}/data:/srv/data -v ${CURRENT_DIR}/src:/srv/src -p 8000:8000 -d --name api_app \
	adzhumurat/leaf_bro_api:latest

push-ui:
	docker build -f services/client/Dockerfile -t adzhumurat/leaf_bro_client . && \
	docker push adzhumurat/leaf_bro_client:latest

run-ui-service:
	docker run -it --rm --network backtier --env-file ${CURRENT_DIR}/.env \
	-p 8501:8501  \
	-v ${CURRENT_DIR}/data:/srv/data \
	-v ${CURRENT_DIR}/services/client/src:/srv/src \
	-e ROOT_DIR=/srv/data \
	-e STREAMLIT_API_URL=api_app \
	--name client_app adzhumurat/leaf_bro_client:latest

run-ui-app:
	ROOT_DIR=$(pwd)/data streamlit run services/client/src/app.py --server.port 8502

run: run-api-app run-ui-service build-network