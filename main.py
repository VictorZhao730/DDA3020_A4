import numpy as np

with open('seeds_dataset.txt', 'r') as file:
    lines = file.readlines()
    data = []
    for line in lines:
        line = line.strip()
        values = line.split('\t')
        new_values = []
        for value in values:
            new_values.append(float(value))
        data.append(new_values)
    data = np.array(data)

def kmeans(data, k):
        # Initialize centroids randomly
    centroids = [data[i] for i in np.random.choice(len(data), k, replace=False)]
    converged = False
    while not converged:
    # Assign data points to nearest centroid
        clusters = [[] for _ in range(k)]
        clusters_index = [[] for _ in range(k)]
        index = 0
        for point in data:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            nearest_centroid = np.argmin(distances)
            clusters[nearest_centroid].append(point)
            clusters_index[nearest_centroid].append(index)
            index = index+1
        # Compute new centroids as mean of points in each cluster
        new_centroids = []
        for cluster in clusters:
            new_centroids.append(np.mean(cluster, axis=0))
        # Check for convergence
        converged = np.array_equal(centroids, new_centroids)
        centroids = new_centroids
    return centroids, clusters, clusters_index

# Example usage
# data = [[1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1], [1, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3, 3]]
centroids, clusters, clusters_index = kmeans(data, 3)
print(clusters_index)

