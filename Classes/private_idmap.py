import numpy as np
import faiss

class PrivateIDMapIndex():
    """
    Private-IDMap index: The Private-IDMap index is a differentially private variant of the IDMap index, which is another commonly used index for approximate nearest neighbor search. The Private-IDMap index adds noise to the similarity scores between the query vector and the indexed vectors to achieve differential privacy. The level of privacy is controlled by the epsilon parameter, which determines the amount of noise added.
    """
    def __init__(self, epsilon=1.0, batch_size=1000):
        self.epsilon = epsilon
        self.batch_size = batch_size
        
    def build_index(self, X):
        # Add noise to the data
        noisy_X = self.add_noise(X)
        
        # Build the IDMap index using the noisy data
        d = X.shape[1]
        index = faiss.IndexIDMap(faiss.IndexFlatL2(d))
        index.add_with_ids(noisy_X, np.arange(X.shape[0]))
        
        return index
    
    def add_noise(self, X):
        # Compute the standard deviation of the noise
        sigma = np.sqrt(2 * np.log(1.25 / self.epsilon)) * np.linalg.norm(X, ord='fro') / X.shape[0]
        
        # Batch the input data
        num_batches = int(np.ceil(X.shape[0] / self.batch_size))
        noisy_X_batches = []
        for i in range(num_batches):
            # Get the current batch of data
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, X.shape[0])
            X_batch = X[start_idx:end_idx]
            
            # Compute the noisy batch of data
            noise = np.random.normal(loc=0, scale=sigma, size=X_batch.shape)
            noisy_X_batch = X_batch + noise
            
            # Append the noisy batch to the list of noisy batches
            noisy_X_batches.append(noisy_X_batch)
        
        # Concatenate the batches back into a single array
        noisy_X = np.concatenate(noisy_X_batches, axis=0)
        
        return noisy_X
