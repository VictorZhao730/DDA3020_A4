import numpy as np
from scipy.stats import multivariate_normal

def kmeans(data, k, max_iter=100):
    # Initialize centroids randomly
    n_samples, n_features = data.shape
    centroids = data[np.random.choice(n_samples, k, replace=False)]
    for _ in range(max_iter):
        # Assign data points to closest centroids
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        clusters = np.argmin(distances, axis=0)
        # Update centroids to mean of data points in each cluster
        new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters

# Load seed dataset from file
data = np.loadtxt('seeds_dataset.txt')
# Extract features and labels from data
X = data[:, :-1]
y = data[:, -1]
# Run k-means clustering with k=3
centroids, clusters = kmeans(X, k=3)



def gmm_em(data, k, max_iter=100):
    n_samples, n_features = data.shape
    # Initialize means and covariances randomly
    means = data[np.random.choice(n_samples, k, replace=False)]
    covariances = [np.eye(n_features)] * k
    # Initialize weights uniformly
    weights = np.ones(k) / k
    for _ in range(max_iter):
        # E-step: compute responsibilities
        responsibilities = []
        for i in range(k):
            responsibility = weights[i] * multivariate_normal(mean=means[i], cov=covariances[i]).pdf(data)
            responsibilities.append(responsibility)
        responsibilities = np.array(responsibilities).T
        responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]
        # M-step: update means, covariances, and weights
        new_means = []
        new_covariances = []
        new_weights = []
        for i in range(k):
            # Update means
            weighted_sum = (responsibilities[:, i][:, np.newaxis] * data).sum(axis=0)
            new_mean = weighted_sum / responsibilities[:, i].sum()
            new_means.append(new_mean)
            # Update covariances
            diff = data - new_mean
            weighted_sum = (responsibilities[:, i][:, np.newaxis] * diff[:, :, np.newaxis] * diff[:, np.newaxis, :]).sum(axis=0)
            new_covariance = weighted_sum / responsibilities[:, i].sum()
            new_covariances.append(new_covariance)
            # Update weights
            new_weight = responsibilities[:, i].mean()
            new_weights.append(new_weight)
        means = new_means
        covariances = new_covariances
        weights = new_weights
    return means, covariances, weights

# Load seed dataset from file
data = np.loadtxt('seeds_dataset.txt')
# Extract features and labels from data
X = data[:, :-1]
y = data[:, -1]
# Run GMM-EM clustering with k=3
means, covariances, weights = gmm_em(X, k=3)


def silhouette_coefficient(data, labels):
    n_samples = data.shape[0]
    silhouette_vals = np.zeros(n_samples)
    for i in range(n_samples):
        # Calculate average distance to other points in same cluster
        same_cluster = data[labels == labels[i]]
        same_cluster = same_cluster[same_cluster != data[i]]
        a = np.linalg.norm(same_cluster - data[i], axis=1).mean()
        # Calculate average distance to points in next nearest cluster
        nearest_cluster = data[labels != labels[i]]
        b = np.linalg.norm(nearest_cluster - data[i], axis=1).mean()
        # Calculate silhouette value for point i
        silhouette_vals[i] = (b - a) / max(a, b)
    # Return average silhouette value for all points
    return silhouette_vals.mean()

# Load seed dataset from file
data = np.loadtxt('seeds_dataset.txt')
# Extract features and labels from data
X = data[:, :-1]
y = data[:, -1]
# Run k-means clustering with k=3
centroids, clusters = kmeans(X, k=3)
# Calculate silhouette coefficient for k-means clustering result
silhouette_coef = silhouette_coefficient(X, clusters)
print(f'Silhouette coefficient for k-means clustering: {silhouette_coef:.3f}')



def rand_index(labels1, labels2):
    n_samples = labels1.shape[0]
    # Calculate number of pairs of points with same labels
    same_labels = (labels1 == labels2).sum()
    # Calculate number of pairs of points with different labels
    different_labels = n_samples * (n_samples - 1) / 2 - same_labels
    # Calculate Rand index
    index = (same_labels + different_labels) / (n_samples * (n_samples - 1) / 2)
    return index

# Load seed dataset from file
data = np.loadtxt('seeds_dataset.txt')
# Extract features and labels from data
X = data[:, :-1]
y = data[:, -1]
# Run k-means clustering with k=3
centroids, clusters = kmeans(X, k=3)
# Calculate Rand index between k-means clustering result and true labels
rand_idx = rand_index(clusters, y)
print(f'Rand index for k-means clustering: {rand_idx:.3f}')





# Load seed dataset from file
data = np.loadtxt('seeds_dataset.txt')
# Extract features and labels from data
X = data[:, :-1]
y = data[:, -1]

n_trials = 10
silhouette_coefs = np.zeros(n_trials)
rand_idxs = np.zeros(n_trials)
for i in range(n_trials):
    # Run k-means clustering with k=3
    centroids, clusters = kmeans(X, k=3)
    # Calculate silhouette coefficient for k-means clustering result
    silhouette_coefs[i] = silhouette_coefficient(X, clusters)
    # Calculate Rand index between k-means clustering result and true labels
    rand_idxs[i] = rand_index(clusters, y)

# Calculate standard deviations of silhouette coefficients and Rand indexes
silhouette_coef_std = silhouette_coefs.std()
rand_idx_std = rand_idxs.std()

print(f'Standard deviation of silhouette coefficients for k-means clustering: {silhouette_coef_std:.3f}')
print(f'Standard deviation of Rand indexes for k-means clustering: {rand_idx_std:.3f}')





# Load seed dataset from file
data = np.loadtxt('seeds_dataset.txt')
# Extract features and labels from data
X = data[:, :-1]
y = data[:, -1]

n_trials = 10
silhouette_coefs = np.zeros(n_trials)
rand_idxs = np.zeros(n_trials)
for i in range(n_trials):
    # Run GMM-EM clustering with k=3
    means, covariances, weights = gmm_em(X, k=3)
    # Calculate clusters based on maximum likelihood
    responsibilities = []
    for j in range(3):
        responsibility = weights[j] * multivariate_normal(mean=means[j], cov=covariances[j]).pdf(X)
        responsibilities.append(responsibility)
    responsibilities = np.array(responsibilities).T
    clusters = np.argmax(responsibilities, axis=1)
    # Calculate silhouette coefficient for GMM-EM clustering result
    silhouette_coefs[i] = silhouette_coefficient(X, clusters)
    # Calculate Rand index between GMM-EM clustering result and true labels
    rand_idxs[i] = rand_index(clusters, y)

# Calculate standard deviations of silhouette coefficients and Rand indexes
silhouette_coef_std = silhouette_coefs.std()
rand_idx_std = rand_idxs.std()

print(f'Standard deviation of silhouette coefficients for GMM-EM clustering: {silhouette_coef_std:.3f}')
print(f'Standard deviation of Rand indexes for GMM-EM clustering: {rand_idx_std:.3f}')
