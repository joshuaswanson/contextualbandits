class Distribution:
    pass

class Gaussian(Distribution):
    
    def __init__(self, theta, V, S=0):
        super().__init__()
        self.theta = theta
        self.V = V
        self.S = S

    
    def update_posterior(self, x, y, copy=False):
        if copy:
            V = self.V + np.outer(x, x)
            S = self.S + x * y
            theta = np.linalg.inv(V) @ S
            return Gaussian(theta, V, S)
        else:
            self.V += np.outer(x, x)
            self.S += x * y
            self.theta = np.linalg.inv(self.V) @ self.S
    
    
    def sample():
        def f(x):
            return x @ np.random.multivariate_normal(self.theta, self.V)
        return f