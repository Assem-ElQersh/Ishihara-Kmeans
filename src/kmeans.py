import numpy as np

class KMeansClustering:
    """
    Implementation of K-means clustering algorithm from scratch
    """
    def __init__(self, k=3, max_iterations=100, random_state=42):
        """
        Initialize K-means with parameters
        
        Args:
            k (int): Number of clusters
            max_iterations (int): Maximum number of iterations
            random_state (int): Random seed for reproducibility
        """
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        np.random.seed(random_state)
        
    def initialize_centroids(self, X):
        """
        Initialize centroids randomly from the data points
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Initial centroids
        """
        # Get the number of samples and features
        n_samples, n_features = X.shape
        
        # Initialize centroids array
        centroids = np.zeros((self.k, n_features))
        
        # Randomly select k data points as initial centroids
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        centroids = X[random_indices]
        
        return centroids
    
    def compute_distance(self, X, centroids):
        """
        Compute Euclidean distance between each point and each centroid
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            centroids (numpy.ndarray): Centroids of shape (k, n_features)
            
        Returns:
            numpy.ndarray: Distances of shape (n_samples, k)
        """
        # Get the number of samples
        n_samples = X.shape[0]
        
        # Initialize distances array
        distances = np.zeros((n_samples, self.k))
        
        # Compute Euclidean distance for each sample to each centroid
        for i in range(self.k):
            # Vectorized computation of Euclidean distance
            distances[:, i] = np.sqrt(np.sum((X - centroids[i])**2, axis=1))
            
        return distances
    
    def assign_clusters(self, distances):
        """
        Assign each data point to the closest centroid
        
        Args:
            distances (numpy.ndarray): Distances of shape (n_samples, k)
            
        Returns:
            numpy.ndarray: Cluster labels of shape (n_samples,)
        """
        # Assign each point to the cluster with minimum distance
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X, labels):
        """
        Update centroids based on mean of points in each cluster
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            labels (numpy.ndarray): Cluster labels of shape (n_samples,)
            
        Returns:
            numpy.ndarray: Updated centroids
        """
        # Get the number of features
        n_features = X.shape[1]
        
        # Initialize new centroids
        new_centroids = np.zeros((self.k, n_features))
        
        # Update centroids by computing mean of points in each cluster
        for i in range(self.k):
            # Get points assigned to cluster i
            cluster_points = X[labels == i]
            
            # Handle empty clusters by keeping the previous centroid
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # If cluster is empty, initialize with a random point
                new_centroids[i] = X[np.random.randint(X.shape[0])]
                
        return new_centroids
    
    def has_converged(self, old_centroids, new_centroids, tolerance=1e-4):
        """
        Check if centroids have converged
        
        Args:
            old_centroids (numpy.ndarray): Old centroids
            new_centroids (numpy.ndarray): New centroids
            tolerance (float): Convergence tolerance
            
        Returns:
            bool: True if converged, False otherwise
        """
        # Compute Euclidean distance between old and new centroids
        distances = np.sqrt(np.sum((new_centroids - old_centroids)**2, axis=1))
        
        # Check if all distances are below tolerance
        return np.all(distances < tolerance)
    
    def fit(self, X):
        """
        Fit K-means model to data
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            self
        """
        # Initialize centroids
        self.centroids = self.initialize_centroids(X)
        
        # Run K-means algorithm
        for _ in range(self.max_iterations):
            # Compute distances
            distances = self.compute_distance(X, self.centroids)
            
            # Assign clusters
            labels = self.assign_clusters(distances)
            
            # Store old centroids for convergence check
            old_centroids = self.centroids.copy()
            
            # Update centroids
            self.centroids = self.update_centroids(X, labels)
            
            # Check for convergence
            if self.has_converged(old_centroids, self.centroids):
                break
                
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Predicted cluster labels
        """
        # Compute distances
        distances = self.compute_distance(X, self.centroids)
        
        # Assign clusters
        return self.assign_clusters(distances)