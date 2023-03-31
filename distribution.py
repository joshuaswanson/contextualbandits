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
    def __init__(self, theta, V, Vinv=None, S=0, sigma=1):
        super().__init__()
        self.theta = theta
        self.V = V
        self.S = S
        self.sigma = sigma
        if Vinv is None:
            self.Vinv = np.linalg.inv(V)
        else:
            self.Vinv = Vinv

    
    def update_posterior(self, x, y, copy=False):
        if copy:
            V = self.V + np.outer(x, x)
            S = self.S + np.dot(x, y)
            theta = np.linalg.inv(V) @ S
            return Gaussian(theta, V, Vinv=np.linalg.inv(V), S=S)
        else:
            self.V += np.outer(x, x)
            self.S += x * y
            self.Vinv = np.linalg.inv(self.V)
            self.theta = self.Vinv @ self.S
    
    
    def sample(self, k=1):
        if k==1:
            theta_tilde = np.random.multivariate_normal(self.theta, self.Vinv)
        else:
            theta_tilde = np.random.multivariate_normal(self.theta, self.Vinv, size=k)
        # def f(x):
        #     return x @ theta_tilde
        if k==1:
            return GenericFunction(lambda x: x@theta_tilde, self.sigma)

        return [GenericFunction(lambda x: x@theta.T, self.sigma) for theta in theta_tilde] 
    
    def map(self):
        return GenericFunction(lambda x: x@self.theta, self.sigma)