import numpy as np
import faiss


class PrivateLSHIndex:
    """
    Private-LSH index: The Private-LSH index is a differentially private variant of the LSH index, which is a family of hash functions commonly used for approximate nearest neighbor search. The Private-LSH index adds noise to the hash function parameters to achieve differential privacy. The level of privacy is controlled by the epsilon parameter, which determines the amount of noise added.
    """
    def __init__(self, d, nlist, nbits, epsilon):
        self.d = d
        self.nlist = nlist
        self.nbits = nbits
        self.epsilon = epsilon
        self.index = None
        
    def add_noise(self, X):
        batch_size = 5000
        n = X.shape[0]
        noise_scale = np.sqrt(2 * np.log(1.25 / self.epsilon)) * np.max(X) / np.sqrt(n)
        noisy_X = np.zeros_like(X)
        for i in range(0, n, batch_size):
            noisy_X[i:i+batch_size] = X[i:i+batch_size] + np.random.normal(scale=noise_scale, size=X[i:i+batch_size].shape)
        return noisy_X
    
    def build_index(self, X):
        noisy_X = self.add_noise(X)
        quantizer = faiss.IndexFlatL2(self.d)
        pca_matrix, _, _ = faiss.PCAMatrix.compute(noisy_X, self.nbits)
        index = faiss.IndexLSH(self.nbits, self.nlist)
        index.train(pca_matrix)
        index.add(pca_matrix)
        self.index = index
        return index
