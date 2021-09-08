from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans

import glob

model = SentenceTransformer('nli-distilroberta-base-v2')

file_list = []

for f in glob.glob('./data/input/*'):
    file_list.append(f)

corpus = []
for file in file_list:
    with open(file, "r") as f:
        corpus = f.read().splitlines()
# corpus = ['The cat sits outside',
#              'A man is playing guitar',
#              'The new movie is awesome']

print(corpus)
corpus_embeddings = model.encode(corpus)

# Perform kmean clustering
num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    print(cluster)
    print("")