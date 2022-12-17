import numpy as np

def kmeans(data, k):
        # Initialize centroids randomly
    centroids = [data[i] for i in np.random.choice(len(data), k, replace=False)]
    converged = False
    while not converged:
    # Assign data points to nearest centroid
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            nearest_centroid = np.argmin(distances)
            clusters[nearest_centroid].append(point)
        # Compute new centroids as mean of points in each cluster
        new_centroids = []
        for cluster in clusters:
            new_centroids.append(np.mean(cluster, axis=0))
        # Check for convergence
        converged = np.array_equal(centroids, new_centroids)
        centroids = new_centroids
    return centroids, clusters

# Example usage
data = [[1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1], [1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3, 3]]
data = np.array(data)
centroids, clusters = kmeans(data, 3)

print(clusters)
"""
This implementation first initializes the centroids randomly from the data points, then iteratively assigns each data point to the nearest centroid and recomputes the centroids as the mean of the points in each cluster. The algorithm stops when the centroids do not change between iterations, indicating convergence.
"""