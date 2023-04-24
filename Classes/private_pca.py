import numpy as np
import faiss

class PrivatePCAIndex():
    """
    Private-PCA index: The Private-PCA index is a differentially private variant of the PCA index, which is commonly used for approximate nearest neighbor search. The Private-PCA index adds noise to the eigenvectors of the PCA decomposition to achieve differential privacy. The level of privacy is controlled by the epsilon parameter, which determines the amount of noise added.
    """
    def __init__(self, epsilon=1.0, batch_size=1000):
        self.epsilon = epsilon
        self.batch_size = batch_size
    
    def build_index(self, X):
        # Add noise to the data in batches
        noisy_X = np.empty_like(X)
        batch_size = self.batch_size
        num_batches = X.shape[0] // batch_size
        remainder = X.shape[0] % batch_size
        
        for i in range(num_batches):
            noisy_X[i*batch_size:(i+1)*batch_size] = self.add_noise(X[i*batch_size:(i+1)*batch_size])
        
        if remainder != 0:
            noisy_X[num_batches*batch_size:] = self.add_noise(X[num_batches*batch_size:])
        
        # Compute the PCA projection matrix
        d = X.shape[1]
        pca_matrix = faiss.PCAMatrix(d, 256)
        pca_matrix.train(noisy_X)
        projection_matrix = faiss.vector_to_array(pca_matrix.PCAMat).reshape(256, d).T
        
        # Transform the data using the projection matrix
        transformed_X = np.dot(X, projection_matrix)
        
        # Build the FAISS index using the transformed data
        index = faiss.IndexFlatL2(256)
        index.add(transformed_X)
        
        return index
    
    def add_noise(self, X):
        # Compute the standard deviation of the noise
        sigma = np.sqrt(2 * np.log(1.25 / self.epsilon)) * np.linalg.norm(X, ord='fro') / X.shape[0]
        
        # Add Gaussian noise to the data
        # noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
        # noisy_X = X + noise
        
        # return noisy_X

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