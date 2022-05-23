import numpy as np
from gensim.models import Word2Vec
from scipy import spatial
from sklearn.cluster import DBSCAN

# Three sentences to compare
sentence1 = "I am testing this word2vec routine by seeing if sentences are related"
sentence2 = "I am attempting to see if this sentence is similar to the previous one using the word2vec routine"
sentence3 = "My morning routine consists of eating crayons and gluing carrots to my face"

# Create array to pass to model
sentences = [sentence1.split(" "), sentence2.split(" "), sentence3.split(" ")]

# Initialise model
model = Word2Vec(sentences, vector_size=10, min_count=1)

# Get vectors
sent1 = np.mean(np.array(model.wv[sentence1.split(" ")]), axis=0)
sent2 = np.mean(np.array(model.wv[sentence2.split(" ")]), axis=0)
sent3 = np.mean(np.array(model.wv[sentence3.split(" ")]), axis=0)

# Create vector array
sentences_t = np.array([sent1, sent2, sent3])

# Check similarity
sim = 1 - spatial.distance.cosine(sent1, sent2)

# Initialise clustering object
db_out = DBSCAN(eps=0.5, min_samples=1, metric=spatial.distance.cosine)

# Output groups
out_arr = db_out.fit_predict(sentences_t)

# Convert results to array output
sentences = np.array([sentence1, sentence2, sentence3])
output = np.vstack((sentences, out_arr)).T
