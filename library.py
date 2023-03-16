import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import multiprocessing as mp


class TopTwoAlgorithm(object):
    def __init__(self, X, theta_star, T, sigma=1, name=""):
        self.X = X
        self.K = X.shape[0]
        self.d = X.shape[1]
        self.theta_star = theta_star
        self.B = np.matmul(X.reshape(-1,self.d,1),
                           X.reshape(-1,1,self.d))
        self.T = T
        self.V = np.eye(self.d)
        self.sigma = sigma
        self.best_x = []
        self.name = name
        
    def run(self, logging_period=1, verbose=False):
        theta = np.zeros(self.d)
        errs = []
        S = 0
        for t in range(self.T):
            theta_1 = np.random.multivariate_normal(theta, self.V)
            best_idx = np.argmax(self.X@theta_1)
            x_1 = self.X[best_idx]
            
            best_idx_2 = best_idx
            while best_idx == best_idx_2:
                theta_2 = np.random.multivariate_normal(theta, self.V)
                best_idx_2 = np.argmax(self.X@theta_2)
                x_2 = self.X[best_idx_2]

            min_idx = np.argmin((x_1 - x_2)@(self.V + self.B)@(x_1 - x_2)) 
            x_n = self.X[min_idx]
            y_n = self.theta_star @ x_n + self.sigma*np.random.randn()

            self.V += np.outer(x_n, x_n)
            S += x_n * y_n
            theta = np.linalg.inv(self.V) @ S
            
            errs.append(np.linalg.norm(theta - self.theta_star))
            
            if t%logging_period == 0:
                # print('run', self.name, 'iter', t,'\n')
                self.best_x.append(np.argmax(self.X@theta))
                if verbose: 
                    plt.xlabel('iteration')
                    plt.ylabel(r'$\|\theta_*-\hat{\theta}\|$', rotation=0, labelpad=30)
                    plt.plot(errs);
                    plt.show()
                    clear_output(wait=True)
    
    def pi(self, theta, V, idx, repeat=10000):
        '''
        Probability that this idx is the best
        '''
        x_star = self.X[idx]
        count = 0        
        for _ in range(repeat):
            random_theta = np.random.multivariate_normal(theta, V)
            count += (idx == np.argmax(X@theta))
        return count / repeat

    @staticmethod
    def gap(self, x, x_star):
        return (x_star -x) @ theta_star 


def A(X, lambda_):
    return X.T@np.diag(lambda_)@X


class ThompsonSampling(object):

    def __init__(self, X, theta_star, T, sigma=1, name=""):
        self.X = X
        self.d = X.shape[1]
        self.sigma = sigma
        self.arm_sequence = []
        self.V = np.eye(self.d)
        self.theta_star = theta_star
        self.best_x = []
        self.name = name
        self.T = T

    def run(self, logging_period=1, verbose=False):
        theta = np.zeros(self.d)
        S = 0
        errs = []
        for t in range(self.T):
            theta_hat = np.random.multivariate_normal(theta, np.linalg.inv(self.V))
            best_idx = np.argmax(self.X @ theta_hat)
            x_n = self.X[best_idx]
            y_n = x_n @ self.theta_star + self.sigma*np.random.randn()
            
            self.V += np.outer(x_n, x_n)
            S += x_n * y_n
            theta = np.linalg.inv(self.V) @ S         
            
            errs.append(np.linalg.norm(theta - self.theta_star))
            
            if t%logging_period == 0:
                # print('run', self.name, 'iter', t,'\n')
                self.best_x.append(np.argmax(self.X@theta))
                if verbose: 
                    plt.xlabel('iteration')
                    plt.ylabel(r'$\|\theta_*-\hat{\theta}\|$', rotation=0, labelpad=30)
                    plt.plot(errs);
                    plt.show()
                    clear_output(wait=True)
                    
class XYStatic(object):
    
    def __init__(self):
        
        
    def run(self):
        
        