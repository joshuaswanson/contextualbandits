import numpy as np

def sphere(K, d):
    '''
    Generates arms and optimal arm of a unit sphere instance.
    K: int, number of arms
    d: int, dimension of each arm
    Returns:
    - X: numpy array, matrix of K arms of dimension d.
    - theta_star: numpy array, 1 x d vector representing the optimal arm.
    '''
    X = np.random.randn(K, d)/np.sqrt(d)
    norms = np.linalg.norm(X, axis=1).reshape(K, 1)
    X /= norms
    theta_star = np.random.randn(d)
    theta_star = theta_star/np.linalg.norm(theta_star)
    return X, theta_star

def soare(d, alpha):
    '''
    Generates arms and optimal arm of a "Soare" instance.
    d: int, dimension of each arm
    alpha: float, 
    Returns:
    - X: numpy array, matrix of d+1 arms of dimension d.
    - 2*e_1: numpy array, 1 x d vector representing the optimal arm.
    '''
    X = np.eye(d)
    e_1, e_2 = X[:2]
    x_prime = np.cos(alpha) * e_1 + np.sin(alpha) * e_2
    X = np.concatenate([X, np.array([x_prime])])
    return X, 2 * e_1