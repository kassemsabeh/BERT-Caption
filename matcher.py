from captions import predict_caption
from os import listdir
import numpy as np
from sentence_transformers import SentenceTransformer
import scipy
from PIL import Image

image_dir = 'Images'
images = listdir(image_dir)
filenames = ['Images/' + image for image in images]

#captions = [predict_caption(file) for file in filenames]

model = SentenceTransformer('bert-base-nli-mean-tokens')
sentence_embeddings = model.encode(captions)
#print(len(captions))
def match_query(query):
    query_embedding = model.encode([query])
    distance = scipy.spatial.distance.cdist(query_embedding, sentence_embeddings, 'cosine')[0]
    results = zip(range(len(distance)), distance)
    results = sorted(results, key=lambda x:x[1])
    for i, distance in results[:1]:
        print(captions[i].strip(), 'cosine score: %4.f' %(1 - distance))
        print(f"Image  name: {filenames[i]}")
        #caption = captions[i].strip()
        #score = 1 - distance
        image = Image.open(filenames[i])

        return image

