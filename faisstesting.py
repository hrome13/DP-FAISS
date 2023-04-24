from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import faiss
import diffprivlib as dp
from diffprivlib.models import GaussianNB

def add_gaussian_noise(data, batch_size=1000, epsilon=0.1):
    # Adds Gaussian noise to data in-place, in batches (to handle huge datasets)

    # Define the mean and standard deviation of the Gaussian noise
    mean = 0
    stddev = 0.1

    # Add Gaussian noise to the embeddings in batches
    for i in range(0, data.shape[0], batch_size):
        batch_end = min(i+batch_size, data.shape[0])
        batch_size_actual = batch_end - i
        noise = np.random.normal(mean, stddev, size=(batch_size_actual, data.shape[1]))
        data[i:batch_end] += noise
    
    return data

def add_laplace_noise(data, batch_size=1000, epsilon=1.0):
    # Adds Laplace mechanism noise to data in-place, in batches (to handle huge datasets)

    # Calculate the sensitivity of the data
    sensitivity = np.max(data) - np.min(data)

    # Calculate the scale parameter for the Laplace noise
    scale = sensitivity / epsilon

    # Add Gaussian noise to the embeddings in batches
    for i in range(0, data.shape[0], batch_size):
        batch_end = min(i+batch_size, data.shape[0])
        batch_size_actual = batch_end - i
        noise = np.random.laplace(scale=scale, size=(batch_size_actual, data.shape[1]))
        data[i:batch_end] += noise
    
    return data

# Load the iris dataset
dataset = datasets.load_iris()
original_data = dataset.data.copy()
add_laplace_noise(dataset.data)
# add_gaussian_noise(dataset.data)

# Create an FAISS index
ncentroids = 1024
niter = 20
verbose = True
d = dataset.data.shape[1] # number of dimensions
index = faiss.IndexFlatL2(d) # L2 distance metric
index.add(dataset.data) # put the data in the index
index.nprobe = 10 # number of nearby cells to search
D, I = index.search(np.array([[6, 3, 3, 2]]), 4)
print(I)
for i in I[0]:
    print(original_data[i])

print(index.ntotal)