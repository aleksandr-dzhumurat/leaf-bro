import os

from dagster import asset
from google.cloud import storage

def download_cs_file(bucket_name, file_name, destination_file_name): 
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(file_name)
    blob.download_to_filename(destination_file_name)

    return True

@asset
def load_data() -> None:
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
