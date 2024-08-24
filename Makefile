CURRENT_DIR = $(shell pwd)
NETWORK_NAME = service_network
include .env
export


prepare-dirs:
	mkdir -p ${CURRENT_DIR}/data || true & \
	mkdir -p ${CURRENT_DIR}/data/es-data || true

elastic:
	docker-compose up elasticsearch