import numpy as np

def sphere(K,d):
    X = np.random.randn(K, d)/np.sqrt(d)  # arms
    norms = np.linalg.norm(X, axis=1).reshape(K, 1)
    X = X/norms
    theta_star = np.random.randn(d)
    theta_star = theta_star/np.linalg.norm(theta_star)
    return X, theta_star

def soare(d, alpha):
    X = np.eye(d)
    e_1, e_2 = X[:2]
    x_prime = np.cos(alpha) * e_1 + np.sin(alpha) * e_2
    X = np.concatenate([X, np.array([x_prime])])
    return X, e_1