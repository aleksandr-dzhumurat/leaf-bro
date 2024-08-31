import json
import os
import random
import datetime
from typing import List
from dataclasses import dataclass
import hashlib
import re

import requests
import yaml
import pandas as pd
import backoff
import numpy as np
import openai
from openai import OpenAI

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

def check_csv_schema(documents):
    sample_strain_dict = random.choice(documents)
    check_all_fields_present(sample_strain_dict, Strain)

def get_elastic_client():
    ELASTIC_URL = os.environ['ELASTIC_URL']
    es_client = Elasticsearch(ELASTIC_URL)

    index_url = os.path.join(ELASTIC_URL, config['elastic_index_name'])
    requests.delete(index_url).json()

    return es_client


def download_gcs_file(bucket_name, file_name, destination_file_name): 
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

def load_json(input_file_path):
    with open(input_file_path, 'r', encoding="utf-8") as f:
        reviews_dict = json.load(f)
    return reviews_dict

def get_item_reviews(item_name, reviews_dict):
    reviews = [i['review'].replace('\n', ' ') for i in reviews_dict if i['item_name'] == item_name]
    return reviews

def aggregate_reviews(content_df, reviews_dict):
    res = {}
    for name in content_df['item_name'].unique():
      res[name] = {'reviews': ''.join(get_item_reviews(name, reviews_dict))}
    return res


client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

def get_id_hash(input_text):
    seed_phrase = f'{str(datetime.datetime.now().timestamp())}{input_text}'
    res = str(hashlib.md5(seed_phrase.encode('utf-8')).hexdigest())[:12]
    return res

@backoff.on_exception(backoff.expo, openai.APIError)
@backoff.on_exception(backoff.expo, openai.RateLimitError)
@backoff.on_exception(backoff.expo,openai.Timeout)
@backoff.on_exception(backoff.expo, RuntimeError)
def gpt_query(gpt_params, verbose: bool = False, avoid_fuckup: bool = False) -> dict:
    print('connecting OpenAI...')
    if verbose:
        print(gpt_params["messages"][1]["content"])
    response = client.chat.completions.create(
        **gpt_params
    )
    gpt_response = response.choices[0].message.content
    if avoid_fuckup:
        if '[' in gpt_response or '?' in gpt_response or '{' in gpt_response:
            raise RuntimeError
    res = {'recs': gpt_response}
    res.update({'completion_tokens': response.usage.completion_tokens, 'prompt_tokens': response.usage.prompt_tokens, 'total_tokens': response.usage.total_tokens})
    res.update({'id': get_id_hash(gpt_response)})
    return res

def random_shuffle(input_list) -> str:
    """
    my_list = ['apple', 'banana', 'cherry', 'date', 'fig']
    random_shuffle(my_list)
    """
    random.shuffle(input_list)
    return ', '.join(input_list)

def promt_generation(candidates):
  # TODO: use jinja2
  flavours_list = ['sweet', 'citrus', 'fruity', 'spicy', 'berry', 'pine', 'herbal', 'sour', 'lemon', 'woody', 'grape', 'tropical', 'diesel', 'skunky', 'nutty', 'blueberry', 'vanilla', 'creamy', 'candy', 'cherry']
  reliefs_list = ['depression', 'chronic pain', 'stress', 'insomnia', 'anxiety', 'fatigue', 'nausea', 'migraines', 'muscle spasms', 'headaches', 'ptsd', 'loss of appetite', 'add/adhd', 'inflammation', 'arthritis', 'bipolar disorder', 'cramps', 'pms', 'fibromyalgia', 'gastrointestinal disorder']
  promt = f"""
      generate 5 search query based on strain description below. All queries should be 3-5 words long. Do not include strain name.
      Do not add word "strain".
      Base query on flavours (not connstrained with this list): {random_shuffle(flavours_list)}
      reliefs: {random_shuffle(reliefs_list)}
      effects (body and mind)
      {candidates}
      Queries:
  """
  return promt

def generate(gpt_prompt, verbose=False):
    gpt_params = {
        'model': 'gpt-3.5-turbo',
        'max_tokens': 500,
        'temperature': 0.7,
        'top_p': 0.5,
        'frequency_penalty': 0.5,
    }
    if verbose:
        print(gpt_prompt)
    messages = [
        {
          "role": "system",
          "content": "You are a helpful assistant for medicine shopping",
        },
        {
          "role": "user",
          "content": gpt_prompt,
        },
    ]
    gpt_params.update({'messages': messages})
    res = gpt_query(gpt_params, verbose=False)
    return res

def clean_text(text):
    cleaned_text = re.sub(r'^\d+\.\s*', '', text)
    return cleaned_text

def prepare_ground_truth(queries_dict, data_path):
    ground_truth = []
    for k in queries_dict:
        queries = queries_dict[k]['queries']['recs'].split('\n')
        ground_truth += [{'answer': k, 'query': clean_text(q)} for q in queries]
    print('Num generates: %d' % len(ground_truth))
    with open(data_path, 'w') as f:
      json.dump(ground_truth, f)
    print('data saved to %s' % data_path)
    return ground_truth

@asset(group_name="retrieval_evaluation")
def prepare_queies_for_evaluation():
    root_dir = os.environ['DATA_PATH']
    reviews_path = os.path.join(root_dir, 'pipelines-data', 'content_reviews_green_bro.json')
    res_csv_path = os.path.join(root_dir, 'pipelines-data', 'content_green_bro.csv')
    content_queries_filename = os.path.join(root_dir, 'pipelines-data', 'content_queries.json')

    content_db = pd.read_csv(res_csv_path)
    reviews_dict = load_json(reviews_path)
    reviews_aggregated = aggregate_reviews(content_db, reviews_dict)
   
    if not os.path.exists(content_queries_filename):
        for _, j in reviews_aggregated.items():
            reviews = j['reviews']
            prompt = promt_generation(reviews)
            j.update['queries'] = generate(prompt)
        with open(content_queries_filename, 'w') as f:
            json.dump(reviews_aggregated, f)
    else:
        with open(content_queries_filename, 'r') as f:
            reviews_aggregated = json.load(f)
    print('Data generated %s' % len(reviews_aggregated))
    output_file_path = os.path.join(root_dir, 'pipelines-data','ground_truth.json')
    prepare_ground_truth(reviews_aggregated, output_file_path)
    print('Data saved to  %s' % output_file_path)

def get_pytorch_model(models_dir, model_name='multi-qa-distilbert-cos-v1'):
  from sentence_transformers import SentenceTransformer

  model_path = os.path.join(models_dir, model_name)

  if not os.path.exists(model_path):
      print('huggingface model loading...')
      embedder = SentenceTransformer(model_name)
      embedder.save(model_path)
  else:
      print('pretrained model loading...')
      embedder = SentenceTransformer(model_name_or_path=model_path)
  print('model loadind done')

  return embedder

embedder = None
def get_or_create_embedder(models_dir, model_name):
    global embedder
    if embedder is None:
        embedder = get_pytorch_model(models_dir, model_name)
    return embedder

def normed_vector(v):
    norm = np.sqrt((v * v).sum())
    v_norm = v / norm
    return v_norm

def prepare_catalog(reviews_agregated):
    index = []
    corpus = []
    for key, content in reviews_agregated.items():
        index.append(key)
        corpus.append(content['reviews'])
    return index, corpus

@asset(group_name="retrieval_evaluation")
def download_huggingface_model():
    root_dir = os.environ['DATA_PATH']
    models_dir=os.path.join(root_dir, 'pipelines-data', 'models')
    get_pytorch_model(models_dir, model_name='multi-qa-distilbert-cos-v1')


@asset(group_name="retrieval_evaluation", deps=[download_huggingface_model])
def eval_embeds():
    root_dir = os.environ['DATA_PATH']
    output_path = os.path.join(root_dir, 'pipelines-data', 'models', 'embeds.npy')

    data_path = os.path.join(root_dir, 'pipelines-data', 'content_queries.json')
    with open(data_path, 'r') as f:
      reviews_agregated = json.load(f)
    
    index, corpus = prepare_catalog(reviews_agregated)
    print(len(index), len(corpus))
    models_dir=os.path.join(root_dir, 'pipelines-data', 'models')
    if not os.path.exists(output_path):
        embedder = get_or_create_embedder(models_dir, model_name='multi-qa-distilbert-cos-v1')
        embeds = embedder.encode(corpus, show_progress_bar=True)
        print(embeds.shape)
        np.save(output_path, embeds)
    data_path = os.path.join(models_dir, 'embeds_index.json')
    with open(data_path, 'w') as f:
      json.dump(index, f)

class VectorSearchEngine:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=10):
        scores = self.embeddings.dot(v_query)
        idx = np.argsort(-scores)[:num_results]
        return [{'score': scores[i], 'doc': self.documents[i]} for i in idx]

def eval_hitrate(golden_dataset):
    root_dir = os.environ['DATA_PATH']
    models_dir = os.path.join(root_dir, 'pipelines-data', 'models')

    index_file_path = os.path.join(models_dir, 'embeds_index.json')
    embeds_file_path = os.path.join(models_dir, 'embeds.npy')
    with open(index_file_path, 'r') as f:
        index = json.load(f)
    embeds = np.load(embeds_file_path)
    print(embeds.shape, len(index))

    search_engine = VectorSearchEngine(documents=index, embeddings=embeds)
    embedder = get_or_create_embedder(models_dir, model_name='multi-qa-distilbert-cos-v1')
    res = []
    for entry in golden_dataset:
        v = embedder.encode(entry['query'])
        res.append( 1 if sum(entry['origin']==i['doc'] for i in search_engine.search(v, num_results=30)) > 0 else 0)
    return sum(res), len(res)

@asset(group_name="retrieval_evaluation", deps=[eval_embeds, prepare_queies_for_evaluation])
def eval_embedder_hitrate():
    root_dir = os.environ['DATA_PATH']
    data_path = os.path.join(root_dir, 'pipelines-data', 'golden_dataset.json')
    with open(data_path, 'r') as f:
        golden_dataset = json.load(f)
    hits, entries = eval_hitrate(golden_dataset)
    print('Hit rate: %.3f, num entries: %d' % (hits/entries, entries))


@asset(group_name="data_ingestion")
def get_content_data() -> None:
    gcs_bucket = os.environ['GCS_BUCKET']
    root_dir = os.environ['DATA_PATH']
    result_filename = os.path.join(root_dir, 'pipelines-data', config['content_file_name'])
    print('Data loading to %s' % result_filename)
    if not os.path.exists(result_filename):
        download_gcs_file(
            bucket_name=gcs_bucket, file_name=config['content_file_name'],
            destination_file_name=result_filename
        )

@asset(group_name="data_ingestion")
def get_reviews_data() -> None:
    gcs_bucket = os.environ['GCS_BUCKET']
    root_dir = os.environ['DATA_PATH']
    result_filename = os.path.join(root_dir, 'pipelines-data', config['reviews_file_name'])
    print('Data loading to %s' % result_filename)
    if not os.path.exists(result_filename):
        download_gcs_file(
            bucket_name=gcs_bucket, file_name=config['reviews_file_name'],
            destination_file_name=result_filename
        )

def read_csv_as_dicts() -> List:
    root_dir = os.environ['DATA_PATH']
    result_filename = os.path.join(root_dir, 'pipelines-data', config['content_file_name'])
    df = pd.read_csv(result_filename)
    df['category'] = 'flower'
    csv_entries = df.to_dict(orient='records')
    print('Num entries: %d' % len(csv_entries))

    return csv_entries

@asset(group_name="data_ingestion", deps=[get_content_data, get_reviews_data])
def elastic_data_ingestion() -> None:
    es_client = get_elastic_client()
    index_entries = read_csv_as_dicts()
    check_csv_schema(index_entries)
    build_es_index(es_client, index_entries, config['elastic_index_name'])
