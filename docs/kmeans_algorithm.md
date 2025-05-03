# K-means Clustering Algorithm Implementation

This document explains the K-means clustering algorithm and its implementation in this project for Ishihara test analysis.

## What is K-means Clustering?

K-means is an unsupervised machine learning algorithm that groups data points into k distinct clusters based on similarity. The algorithm identifies k centroids in the data and assigns each data point to the nearest centroid, creating clusters of similar points.

For Ishihara test images, K-means helps separate pixels into distinct groups based on color similarity, which can help extract the hidden number from the background.

## Algorithm Steps

The K-means algorithm follows these steps:

1. **Initialization**: Randomly select k data points as initial centroids
2. **Assignment**: Assign each point to the nearest centroid, forming k clusters
3. **Update**: Recalculate the centroids by taking the mean of all points in each cluster
4. **Repeat**: Iterate steps 2 and 3 until the centroids no longer change significantly

## Our Implementation

Our implementation in `kmeans.py` provides a complete K-means clustering algorithm from scratch. Here's a breakdown of the key components:

### KMeansClustering Class

This class encapsulates the K-means algorithm with the following methods:

#### Initialization

```python
def __init__(self, k=3, max_iterations=100, random_state=42):
    self.k = k
    self.max_iterations = max_iterations
    self.random_state = random_state
    self.centroids = None
    np.random.seed(random_state)
```

This constructor sets the number of clusters `k`, maximum iterations, and a random seed for reproducibility.

#### Centroid Initialization

```python
def initialize_centroids(self, X):
    n_samples, n_features = X.shape
    centroids = np.zeros((self.k, n_features))
    random_indices = np.random.choice(n_samples, self.k, replace=False)
    centroids = X[random_indices]
    return centroids
```

This method randomly selects k data points from the input data as initial centroids.

#### Distance Computation

```python
def compute_distance(self, X, centroids):
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, self.k))
    for i in range(self.k):
        distances[:, i] = np.sqrt(np.sum((X - centroids[i])**2, axis=1))
    return distances
```

This method computes the Euclidean distance between each data point and each centroid.

#### Cluster Assignment

```python
def assign_clusters(self, distances):
    return np.argmin(distances, axis=1)
```

This method assigns each data point to the cluster with the minimum distance.

#### Centroid Update

```python
def update_centroids(self, X, labels):
    n_features = X.shape[1]
    new_centroids = np.zeros((self.k, n_features))
    for i in range(self.k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            new_centroids[i] = np.mean(cluster_points, axis=0)
        else:
            new_centroids[i] = X[np.random.randint(X.shape[0])]
    return new_centroids
```

This method updates the centroids by computing the mean of all points in each cluster.

#### Convergence Check

```python
def has_converged(self, old_centroids, new_centroids, tolerance=1e-4):
    distances = np.sqrt(np.sum((new_centroids - old_centroids)**2, axis=1))
    return np.all(distances < tolerance)
```

This method checks if the algorithm has converged by measuring the change in centroids.

#### Model Fitting

```python
def fit(self, X):
    self.centroids = self.initialize_centroids(X)
    for _ in range(self.max_iterations):
        distances = self.compute_distance(X, self.centroids)
        labels = self.assign_clusters(distances)
        old_centroids = self.centroids.copy()
        self.centroids = self.update_centroids(X, labels)
        if self.has_converged(old_centroids, self.centroids):
            break
    return self
```

This method fits the K-means model to the data by iterating through the algorithm steps until convergence or maximum iterations.

#### Prediction

```python
def predict(self, X):
    distances = self.compute_distance(X, self.centroids)
    return self.assign_clusters(distances)
```

This method assigns new data points to clusters based on the fitted centroids.

## Application to Ishihara Tests

For Ishihara color blindness test images, we apply K-means clustering as follows:

1. **Preprocessing**: Convert the image to a suitable color space (like Lab) and possibly isolate a specific channel (like the 'a' channel)
2. **Clustering**: Apply K-means to group pixels with similar colors together
3. **Segmentation**: Create a segmented image where each pixel is assigned its cluster's color
4. **Number Extraction**: Identify which cluster(s) represent the hidden number and create a binary mask

## Optimizing K-means for Ishihara Tests

Several factors affect the performance of K-means for Ishihara tests:

### Number of Clusters (k)

- **k=2**: Often sufficient for simple Ishihara tests with clear contrast between number and background
- **k=3**: Better for tests with more complex color patterns
- **k=4+**: Rarely needed, but can be useful for very complex tests

### Initialization Strategy

We use random initialization, but more sophisticated methods like k-means++ could be implemented for potentially better results.

### Convergence Criteria

We use a small tolerance (1e-4) to determine convergence, which provides a good balance between accuracy and speed.

## Performance Considerations

The time complexity of K-means is O(t * k * n * d), where:
- t is the number of iterations
- k is the number of clusters
- n is the number of data points
- d is the number of features

For Ishihara test images, this is typically very manageable since:
- k is small (usually 2-3)
- Convergence usually happens within 10-20 iterations
- We often use only a single channel, reducing d to 1

## References

- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In Proceedings of the fifth Berkeley symposium on mathematical statistics and probability.
