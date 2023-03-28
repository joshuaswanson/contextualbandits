import numpy as np

class Distribution:
    pass


class GenericFunction():
    def __init__(self, f, sigma):
        self.f = f
        self.sigma = sigma

    def pull(self, x):
        """
        Evaluate the function f at x and add random noise.
        """
        noise = np.random.randn(*x.shape[:-1], 1) * self.sigma
        return self.evaluate(x) + noise.squeeze(axis=-1)

    def evaluate(self, x):
        """
        Evaluate the function f at x.
        """
        return self.f(x)



class Gaussian(Distribution):
    def __init__(self, theta, V, S=0, sigma=1):
        super().__init__()
        self.theta = theta
        self.V = V
        self.S = S
        self.sigma = sigma

    
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
    
    
    def sample(self, k=1):
        if k==1:
            theta_tilde = np.random.multivariate_normal(self.theta, self.V)
        else:
            theta_tilde = np.random.multivariate_normal(self.theta, self.V, size=k)
        # def f(x):
        #     return x @ theta_tilde
        if k==1:
            return GenericFunction(lambda x: x@theta_tilde, self.sigma)

        return [GenericFunction(lambda x: x@theta.T, self.sigma) for theta in theta_tilde] 
    
    def map(self):
        return GenericFunction(lambda x: x@self.theta, self.sigma)