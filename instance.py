import numpy as np

def sphere(K,d):
    '''
    K: arms
    d: dimensions
    '''
    X = np.random.randn(K, d)/np.sqrt(d)  # arms
    norms = np.linalg.norm(X, axis=1).reshape(K, 1)
    X /= norms
    theta_star = np.random.randn(d)
    theta_star = theta_star/np.linalg.norm(theta_star)
    return X, theta_star

def soare(d, alpha):
    '''    
    d: dimensions
    alpha: angle
    '''
    X = np.eye(d)
    e_1, e_2 = X[:2]
    x_prime = np.cos(alpha) * e_1 + np.sin(alpha) * e_2
    X = np.concatenate([X, np.array([x_prime])])
    return X, 2 * e_1