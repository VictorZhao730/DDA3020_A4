# Certainly! Here is another implementation of the GMM-EM algorithm for clustering a seven-dimensional dataset into 3 clusters. This implementation is slightly different from the previous one in that it separates the E-step and M-step into separate functions, which may make the code easier to understand and modify.

# First, let's start by importing the necessary libraries and defining some helper functions:

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
# Next, let's define the E-step function, which computes the responsibilities for each sample:

def E_step(data, pi, mu, sigma):
    # Compute the responsibilities
    n = data.shape[0]
    k = mu.shape[0]
    responsibilities = np.zeros((n, k))
    for j in range(k):
        responsibilities[:, j] = pi[j] * multivariate_normal_pdf(data, mu[j], sigma[j])
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities
# Then, let's define the M-step function, which updates the model parameters based on the responsibilities:

def M_step(data, responsibilities):
    # Update the model parameters
    n, d = data.shape
    k = responsibilities.shape[1]
    pi = responsibilities.mean(axis=0)
    mu = (responsibilities[:, :, np.newaxis] * data[:, np.newaxis, :]).sum(axis=0) / responsibilities.sum(axis=0)[:, np.newaxis]
    sigma = np.zeros((k, d, d))
    for j in range(k):
        sigma[j] = ((responsibilities[:, j, np.newaxis] * (data - mu[j])[:, np.newaxis, :]).T @ (data - mu[j])[:, np.newaxis, :]) / responsibilities[:, j].sum()
    return pi, mu, sigma

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
        responsibilities = E_step(data, pi, mu, sigma)

        # M-step: update the model parameters
        pi, mu, sigma = M_step(data, responsibilities)

        # Calculate the log-likelihood of the data given the model parameters
        log_likelihood = log_likelihood(data, pi, mu, sigma)

        # Check for convergence
        if np.abs(log_likelihood - log_likelihood_prev) < tolerance:
            break
        log_likelihood_prev = log_likelihood

    # Return the model parameters
    return pi, mu, sigma
