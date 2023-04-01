import numpy as np

class Linear():
    
    def __init__(self, X, Y, theta_star, T, sigma=1, name=""):
        
        self.X = X
        self.Y = Y
        self.n, self.d = X.shape
        self.theta_star = theta_star
        
        self.V = np.eye(self.d)

        self.T = T  # time steps
        self.sigma = sigma
        self.name = name
        self.arms_recommended = []


class Concept():
    
    def __init__(self, X, gen_star, T, sigma=1, name=""):
        
        self.X = X
        self.n = X.shape[0]
        self.gen_star = gen_star
        
        self.T = T  # time steps
        self.sigma = sigma
        self.name = name
        self.arms_recommended = []