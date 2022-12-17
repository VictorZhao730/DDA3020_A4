"""
Sure! Here is a step-by-step implementation of the GMM-EM algorithm for clustering a seven-dimensional dataset into 3 clusters.

First, let's start by importing the necessary libraries and defining some helper functions:
"""

import numpy as np

# Function to calculate the probability density function (PDF) of a multivariate normal distribution
def multivariate_normal_pdf(x, mu, sigma):
    n = x.shape[0]
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    x_mu = x - mu
    term1 = 1. / (((2*np.pi)**(n/2)) * (sigma_det**(1./2)))
    term2 = (-1./2) * ((x_mu).T @ sigma_inv @ (x_mu))
    return float(term1 * np.exp(term2))

# Function to calculate the log-likelihood of the data given the model parameters
def log_likelihood(data, pi, mu, sigma):
    n = data.shape[0]
    k = mu.shape[0]
    log_likelihood = 0
    for i in range(n):
        log_likelihood_i = 0
        for j in range(k):
            log_likelihood_i += pi[j] * multivariate_normal_pdf(data[i], mu[j], sigma[j])
        log_likelihood += np.log(log_likelihood_i)
    return log_likelihood

def GMM_EM(data, k, max_iter=100, tolerance=1e-4):
    # Initialize the model parameters
    n, d = data.shape
    pi = np.ones(k) / k  # Weights
    mu = data[np.random.choice(range(n), k, replace=False)]  # Means
    sigma = [np.eye(d)] * k  # Covariances

    # Initialize the log-likelihood
    log_likelihood_prev = -np.inf

    # Run the EM algorithm until convergence or until the maximum number of iterations is reached
    for i in range(max_iter):
        # E-step: compute the responsibilities
        responsibilities = np.zeros((n, k))
        for j in range(k):
            responsibilities[:, j] = pi[j] * multivariate_normal_pdf(data, mu[j], sigma[j])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # M-step: update the model parameters
        pi = responsibilities.mean(axis=0)
        mu = (responsibilities[:, :, np.newaxis] * data[:, np.newaxis, :]).sum(axis=0) / responsibilities.sum(axis=0)[:, np.newaxis]
        sigma = np.zeros((k, d, d))
        for j in range(k):
            sigma[j] = ((responsibilities[:, j, np.newaxis] * (data - mu[j])[:, np.newaxis, :]).T @ (data - mu[j])[:, np.newaxis, :]) / responsibilities[:, j].sum()

        # Calculate the log-likelihood of the data given the model parameters
        log_likelihood = log_likelihood(data, pi, mu, sigma)

        # Check for convergence
        if np.abs(log_likelihood - log_likelihood_prev) < tolerance:
            break
        log_likelihood_prev = log_likelihood

    # Return the model parameters
    return pi, mu, sigma

# Generate some fake data
data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=100)

# Run the GMM-EM algorithm with k=2 components
pi, mu, sigma = GMM_EM(data, k=2)

# Print the model parameters
print(f"Weights: {pi}")
print(f"Means: {mu}")
print(f"Covariances: {sigma}")

