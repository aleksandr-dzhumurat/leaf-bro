import os
import random
from typing import List
from dataclasses import dataclass

import requests
import yaml
import pandas as pd
from dagster import asset
from google.cloud import storage
from tqdm.auto import tqdm
from elasticsearch import Elasticsearch


@dataclass
class Strain:
    title: str
    tags: str
    relief: str
    positive_effects: str
    flavours: str
    avg_rating: float
    num_ratings: int

def get_config(config_path = None):
    if config_path is None:
        if os.getenv("CONFIG_PATH") is None:
            config_path = "config.yml"
        else:
            config_path = os.environ["CONFIG_PATH"]

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


config = get_config()


def check_all_fields_present(data_dict, data_class):
    required_fields = {field.name for field in data_class.__dataclass_fields__.values()}
    dict_keys = set(data_dict.keys())
    missing_fields = required_fields - dict_keys
    if missing_fields:
        raise ValueError(f"Missing fields in dictionary: {missing_fields}")
    else:
        print("All fields are present.")


def download_cs_file(bucket_name, file_name, destination_file_name): 
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.download_to_filename(destination_file_name)
    return True

def build_es_index(es_client, documents, index_name):
    es_client.indices.create(index=index_name, body=config['elastic_index_settings'])

    for doc in tqdm(documents):
        es_client.index(index=index_name, document=doc)
    print('Index created')

@asset
def get_gcs_csv_local() -> None:
    gcs_bucket = os.environ['GCS_BUCKET']
    gcs_file_path = os.environ['GCS_DATA_PATH']
    root_dir = os.environ['DATA_PATH']
    output_file_name = 'content_green_bro.csv'
    result_filename = os.path.join(root_dir, 'pipelines-data', output_file_name)
    print('Data loading to %s' % result_filename)
    if not os.path.exists(result_filename):
        download_cs_file(
            bucket_name=gcs_bucket, file_name=gcs_file_path,
            destination_file_name=result_filename
        )

@asset
def get_gcs_json_local() -> None:
    gcs_bucket = os.environ['GCS_BUCKET']
    gcs_file_path = os.environ['GCS_DATA_PATH']
    root_dir = os.environ['DATA_PATH']
    result_filename = os.path.join(root_dir, 'pipelines-data', config['reviews_local_file_name'])
    print('Data loading to %s' % result_filename)
    if not os.path.exists(result_filename):
        download_cs_file(
            bucket_name=gcs_bucket, file_name=gcs_file_path,
            destination_file_name=result_filename
        )

def read_csv_as_dicts() -> List:
    root_dir = os.environ['DATA_PATH']
    result_filename = os.path.join(root_dir, 'pipelines-data', config['content_local_file_name'])
    df = pd.read_csv(result_filename)
    df['category'] = 'flower'
    csv_entries = df.to_dict(orient='records')
    print('Num entries: %d' % len(csv_entries))

    return csv_entries

def check_csv_schema(documents):
    sample_strain_dict = random.choice(documents)
    check_all_fields_present(sample_strain_dict, Strain)

def get_elastic_client():
    ELASTIC_URL = os.environ['ELASTIC_URL']
    es_client = Elasticsearch(ELASTIC_URL)

    index_url = os.path.join(ELASTIC_URL, config['elastic_index_name'])
    requests.delete(index_url).json()

    return es_client

@asset(deps=[get_gcs_csv_local, get_gcs_json_local])
def build_elastic_index() -> None:
    es_client = get_elastic_client()
    index_entries = read_csv_as_dicts()
    check_csv_schema(index_entries)
    build_es_index(es_client, index_entries, config['elastic_index_name'])
