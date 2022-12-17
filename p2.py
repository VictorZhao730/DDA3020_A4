import numpy as np

def silhouette_coefficient(data, labels):
    # Compute average distance between data points within each cluster
    clusters = [[point for i, point in enumerate(data) if labels[i] == label] for label in set(labels)]
    a = np.array([np.mean([np.linalg.norm(point - other_point) for other_point in cluster]) for cluster in clusters])
    # Compute average distance between data points in different clusters
    b = np.array([np.mean([np.mean([np.linalg.norm(point - other_point) for other_point in clusters[other_cluster]]) for other_cluster in range(len(clusters)) if other_cluster != i]) for i, cluster in enumerate(clusters)])
    # Compute silhouette coefficient for each data point
    s = (b - a) / np.maximum(a, b)
    # Return average silhouette coefficient
    return np.mean(s)

# Example usage
data = [[1, 2], [2, 1], [4, 4], [4, 5], [5, 4], [5, 5]]
labels = [0, 0, 1, 1, 1, 1]
print(silhouette_coefficient(data, labels))

def rand_index(labels1, labels2):
    # Compute number of pairs of points with same labels in both lists
    same = 0
    for i in range(len(labels1)):
        for j in range(len(labels1)):
            if labels1[i] == labels1[j] and labels2[i] == labels2[j]:
                same += 1
    # Compute number of pairs of points with different labels in both lists
    different = 0
    for i in range(len(labels1)):
        for j in range(len(labels1)):
            if labels1[i] != labels1[j] and labels2[i] != labels2[j]:
                different += 1
    # Return Rand Index
    return (same + different) / len(labels1)**2

# Example usage
labels1 = [0, 0, 0, 1, 1, 1]
labels2 = [0, 0, 1, 1, 1, 1]
print(rand_index(labels1, labels2))
