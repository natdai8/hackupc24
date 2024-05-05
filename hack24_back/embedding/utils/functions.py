import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

from PIL import Image
import requests
from io import BytesIO

import torchvision.transforms as transforms
import torchvision.models as models

# Load pre-trained ResNet model
resnet50_model = models.resnet50(weights=True)
resnet50_model = torch.nn.Sequential(*list(resnet50_model.children())[:-1])
resnet50_model.eval()

# Load pre-trained custom model
custom_model = models.resnet50(weights=True)
torch.save(custom_model, 'resnet50_pretrained.pt')
custom_model = torch.load('resnet50_pretrained.pt')
custom_model = torch.nn.Sequential(*list(custom_model.children())[:-1])
custom_model.load_state_dict(torch.load('C:/Users/Natal/hackupc24/hack24_back/embedding/utils/model_weights/custom_resnet50_weights2.h5'))

def preprocess_image(img):
    ''' Resize image '''
    base_width = 64
    wpercent = (base_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))

    return img.resize((base_width, hsize), Image.Resampling.LANCZOS)

def image_to_tensor(image):
    ''' Convert image to tensor feasible by ResNet-50 '''
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)

    return image.unsqueeze(0)

def url_to_Image(url):
  ''' Convert from url to image '''
  response = requests.get(url)
  img = Image.open(BytesIO(response.content))
  return img

def get_embedding(model, image):
    ''' Generate embedding from image '''
    image = preprocess_image(image)
    image_tensor = image_to_tensor(image)
    #image_tensor = image_tensor.to('cuda')
    with torch.no_grad():
         features = model(image_tensor)
         #features = features.cpu()
    embedding = features.squeeze().numpy()
    result = [float(emb) for emb in embedding]

    return result

def generate_new_link(link):
  ''' Generate new link from old link '''
  prefix = "https://sttc-stage-zaraphr.inditex.com"

  parts = link.split("/")
  half_link = parts.index("photos")
  new_url = "/".join(parts[half_link:])
  new_link = prefix + '/' + new_url

  return new_link

def generate_embedding(model, index, row):
  link = row['images']
  embedding = ""
  try:
    image = url_to_Image(link)
    embedding = get_embedding(model, image)
  except:
    link = generate_new_link(link)
    image = url_to_Image(link)
    if image:
      embedding = get_embedding(model, image)

  return embedding

def add_embedding_to_collection(collection, row, index, embedding):
  collection.add(
      embeddings=[embedding],
      documents=[row['images']],
      ids=str(index)
      )

def concat_columns(dataframe, bConcat=False):
  if bConcat:
    row_images = pd.concat([dataframe['IMAGE_VERSION_1'],
                            dataframe['IMAGE_VERSION_2'],
                            dataframe['IMAGE_VERSION_3']],
                           ignore_index=True)
  else:
    dataframe = dataframe.drop(columns=['IMAGE_VERSION_2', 'IMAGE_VERSION_3'])
    row_images = dataframe['IMAGE_VERSION_1']

  dataframe = pd.DataFrame({'images': row_images})
  return dataframe

def preprocess_dataframe(dataframe, bConcat=False, size=100):
  dataframe = dataframe.dropna()
  dataframe = concat_columns(dataframe, bConcat=bConcat)
  dataframe = dataframe[:size]
  return dataframe

def get_data_to_embed():
    url = 'https://github.com/llpfdc/HackUPC2024/blob/main/inditextech_hackupc_challenge_images.csv?raw=true'
    df = pd.read_csv(url)
    df = preprocess_dataframe(df, bConcat=False, size=40000)
    data_to_embed = df[2000:3000]

    return data_to_embed

import chromadb

def initialize_chromadb():
    chroma_client = chromadb.PersistentClient(path="/database")
    resnet50_model_collection = chroma_client.get_or_create_collection(name="resnet50_model_embeddings")
    custom_model_collection = chroma_client.get_or_create_collection(name="custom_model_embeddings")

    data_to_embed = get_data_to_embed()

    # Initialize ResNet-50 collection
    collection = resnet50_model_collection
    model = resnet50_model

    print("start embedding resnet")

    embeddings = []
    failed_links = []
    for index, row in data_to_embed.iterrows():
    # Get embedding list for index in Collection (if exists)
        embedding_list = collection.get(str(index))['documents']

        if embedding_list == []:
            # Embedding not exist in ChromaDB
            embedding = generate_embedding(model, index, row)
            if embedding:
                # Embedding could be generated
                add_embedding_to_collection(collection, row, index, embedding)
                embeddings.append(embedding)
            else:
                # Embedding could not be generated
                failed_links.append(row['images'])
        else:
            # Embedding already exist in ChromaDB
            embedding = embedding_list[0]
            embeddings.append(embedding)

    # Drop rows where images could not get retreated
    data_to_embed = data_to_embed.drop(index=failed_links)
    data_to_embed['embedding'] = embeddings

    print("start embedding custom model")

    # Initialize custom model collection
    collection = custom_model_collection
    model = custom_model

    embeddings = []
    failed_links = []

    for index, row in data_to_embed.iterrows():
        # Get embedding list for index in Collection (if exists)
        embedding_list = collection.get(str(index))['documents']

        if embedding_list == []:
            # Embedding not exist in ChromaDB
            embedding = generate_embedding(model, index, row)
            if embedding:
                # Embedding could be generated
                add_embedding_to_collection(collection, row, index, embedding)
                embeddings.append(embedding)
            else:
                # Embedding could not be generated
                failed_links.append(row['images'])
        else:
            # Embedding already exist in ChromaDB
            embedding = embedding_list[0]
            embeddings.append(embedding)

    # Drop rows where images could not get retreated
    data_to_embed = data_to_embed.drop(index=failed_links)
    data_to_embed['embedding'] = embeddings

    print("finished embeddings")

def get_most_similar_url_images(image):
  ''' Return top 5 most similar urls using ResNet-50 embeddings '''
  ''' Return top 5 most similar urls using custom model embeddings '''
  ''' All in a single array of lenght 10 '''

  chroma_client = chromadb.PersistentClient(path="/database")
  resnet50_model_collection = chroma_client.get_or_create_collection(name="resnet50_model_embeddings")
  custom_model_collection = chroma_client.get_or_create_collection(name="custom_model_embeddings")

  # Generate embedding ResNet-50
  embedding = get_embedding(resnet50_model, image)
  embedding = [float(emb) for emb in embedding]
  # Query ChromaDB
  result = resnet50_model_collection.query(query_embeddings=embedding, n_results=5)
  resnet50_urls = result['documents'][0]

  # Generate embedding custom model
  embedding = get_embedding(custom_model, image)
  embedding = [float(emb) for emb in embedding]
  # Query ChromaDB
  result = custom_model_collection.query(query_embeddings=embedding, n_results=5)
  custom_model_urls = result['documents'][0]

  return resnet50_urls, custom_model_urls


# FUNCTIONS
# initialize_chromadb()
# get_most_similar_url_images(image_url)