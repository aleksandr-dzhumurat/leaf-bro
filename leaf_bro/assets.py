import os
from typing import List

import pandas as pd
from dagster import asset
from google.cloud import storage
from tqdm.auto import tqdm
from elasticsearch import Elasticsearch


def download_cs_file(bucket_name, file_name, destination_file_name): 
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(file_name)
    blob.download_to_filename(destination_file_name)

    return True

def build_es_index(es_client, documents, index_name):
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"} 
            }
        }
    }

    es_client.indices.create(index=index_name, body=index_settings)

    for doc in tqdm(documents):
        es_client.index(index=index_name, document=doc)
    print('Index created')

@asset
def get_local_csv_path() -> None:
    gcs_bucket = os.environ['GCS_BUCKET']
    gcs_file_path = os.environ['GCS_DATA_PATH']
    root_dir = os.environ['DATA_PATH']
    output_file_name = 'content_green_bro.csv'
    result_filename = os.path.join(root_dir, output_file_name)
    print('Data loading to %s' % result_filename)
    if not os.path.exists(result_filename):
        download_cs_file(
            bucket_name=gcs_bucket, file_name=gcs_file_path,
            destination_file_name=result_filename
        )

@asset(deps=[get_local_csv_path])
def index_entries() -> None:
    root_dir = os.environ['DATA_PATH']
    output_file_name = 'content_green_bro.csv'
    result_filename = os.path.join(root_dir, output_file_name)
    df = pd.read_csv(result_filename)
    index_entries = df.to_dict(orient='records')
    print(len(index_entries))
    es_client = Elasticsearch('http://localhost:9200')
    index_name = "course-questions"
    build_es_index(es_client, index_entries, index_name)
