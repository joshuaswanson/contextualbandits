import numpy as np

def sphere(K,d):
    X = np.random.randn(K, d)/np.sqrt(d)  # arms
    norms = np.linalg.norm(X, axis=1).reshape(K, 1)
    X = X/norms
    theta_star = np.random.randn(d)
    theta_star = theta_star/np.linalg.norm(theta_star)
    return X, theta_star