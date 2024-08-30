import os
import json
import numpy as np

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
    print('Loading embed...')
    if embedder is None:
        embedder = get_pytorch_model(models_dir, model_name)
    return embedder

class VectorSearchEngine:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=10):
        scores = self.embeddings.dot(v_query)
        idx = np.argsort(-scores)[:num_results]
        return [{'score': scores[i], 'doc': self.documents[i]} for i in idx]

import pandas as pd

root_dir = os.environ['API_DATA_PATH']
csvfile_path = os.path.join(root_dir, 'pipelines-data', 'content_green_bro.csv')
content_db = pd.read_csv(csvfile_path)

def get_content(content_names_list):
    res = [
        {'title': row['title'], 'explanation': f"{row['tags']}: {row['positive_effects']}"}
        for _, row in content_db[content_db['item_name'].isin(content_names_list)].iterrows()
    ]
    return res

def get_search_instance():
    root_dir = os.environ['API_DATA_PATH']
    models_dir = os.path.join(root_dir, 'pipelines-data', 'models')

    index_file_path = os.path.join(models_dir, 'embeds_index.json')
    embeds_file_path = os.path.join(models_dir, 'embeds.npy')
    with open(index_file_path, 'r') as f:
        index = json.load(f)
    embeds = np.load(embeds_file_path)
    print(embeds.shape, len(index))
    search_engine = VectorSearchEngine(documents=index, embeddings=embeds)
    return search_engine

search_engine = get_search_instance()
embedder = get_or_create_embedder(os.path.join(os.environ['API_DATA_PATH'], 'models'), model_name='multi-qa-distilbert-cos-v1')