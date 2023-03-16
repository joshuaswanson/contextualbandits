import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import multiprocessing as mp


class TopTwoAlgorithm(object):
    def __init__(self, X, theta_star, T, sigma=1, name=""):
        self.X = X
        self.K = X.shape[0]
        self.d = X.shape[1]
        self.theta = np.zeros(self.d)
        self.theta_star = theta_star
        self.B = np.matmul(X.reshape(-1,self.d,1),
                           X.reshape(-1,1,self.d))
        self.T = T
        self.V = np.eye(self.d)
        self.sigma = sigma
        self.arm_sequence = []
        self.name = name
        
    def run(self, logging_period=1, verbose=False):
        errs = []
        S = 0
        for t in range(self.T):
            theta_1 = np.random.multivariate_normal(self.theta, self.V)
            best_idx = np.argmax(self.X@theta_1)
            x_1 = self.X[best_idx]
            
            best_idx_2 = best_idx
            while best_idx == best_idx_2:
                theta_2 = np.random.multivariate_normal(self.theta, self.V)
                best_idx_2 = np.argmax(self.X@theta_2)
                x_2 = self.X[best_idx_2]

            min_idx = np.argmin((x_1 - x_2) @ (self.V + self.B) @ (x_1 - x_2)) 
            x_n = self.X[min_idx]
            y_n = self.theta_star @ x_n + self.sigma*np.random.randn()

            self.V += np.outer(x_n, x_n)
            S += x_n * y_n
            self.theta = np.linalg.inv(self.V) @ S
            
            errs.append(np.linalg.norm(self.theta - self.theta_star))
            
            if t%logging_period == 0:
                # print('run', self.name, 'iter', t,'\n')
                self.arm_sequence.append(np.argmax(self.X @ self.theta))
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

#     @staticmethod
#     def gap(self, x, x_star=self.x_star):
#         return (x_star - x) @ theta_star


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
                self.arm_sequence.append(np.argmax(self.X@theta))
                if verbose: 
                    plt.xlabel('iteration')
                    plt.ylabel(r'$\|\theta_*-\hat{\theta}\|$', rotation=0, labelpad=30)
                    plt.plot(errs);
                    plt.show()
                    clear_output(wait=True)
                    
                    
class XYStatic(object):
    
    def __init__(self, X, theta_star, T, sigma=1, name=""):
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.sigma = sigma
        self.arm_sequence = []
        self.V = np.eye(self.d)
        self.theta_star = theta_star
        self.name = name
        self.T = T
        
        
    def run(self, logging_period=1):
        lam_f = self.frank_wolfe(self.X)
        self.arm_sequence = np.random.choice(self.n, self.T, p=lam_f)
    
    
    def calc_max_mat_norm(self, X, A_inv):
        n = X.shape[0]
        res = np.zeros(n)
        for i in range(n):
            x = X[i]
            res[i] = x.T @ A_inv @ x
        ind = np.argmax(res)
        return res[ind], ind
        
        
    def f(self, X, lam):
        A = X.T @ np.diag(lam) @ X + 0.0001 * np.eye(self.d)
        A_inv = np.linalg.inv(A)
        res, ind = self.calc_max_mat_norm(X, A_inv)
        return res, ind, A_inv

    
    def grad_f(self, X, lam):
        _, ind, A_inv = self.f(X, lam)
        return -np.power(X @ A_inv @ X[ind], 2)
        
        
    def frank_wolfe(self, X, N=500):
        inds = np.random.choice(self.n, 2*self.d, p=1/self.n * np.ones(self.n)).tolist()
        lam = np.bincount(inds, minlength=self.n) / len(inds)
        for i in range(2*self.d+1, N+1):
            eta = 2/(i+1)
            ind = np.argmin(self.grad_f(self.X, lam))
            lam = (1-eta)*lam + eta * np.eye(1, self.n, ind).flatten()
            inds.append(ind)
        return lam
    

class XYAdaptive(object):
    
    def __init__(self, X, theta_star, T, sigma=1, name=""):
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.theta = np.zeros(self.d)
        self.sigma = sigma
        self.arm_sequence = []
        self.V = np.eye(self.d)
        self.theta_star = theta_star
        self.name = name
        self.T = T
        self.k = 20
        # self.k = k #TODO: add this later
        
        
        
    def run(self, logging_period=1):
        S = 0
        for t in range(self.T):
            theta_mat = np.random.multivariate_normal(mean=self.theta, cov=self.V, size=self.k)
            max_x_vec = np.argmax(self.X @ theta_mat.transpose(), axis=0)  # this should be dimension k
            
            X_t = self.X[max_x_vec]
            xy_static = XYStatic(X_t, self.theta_star, 1, sigma=1, name="")
            lam_f = xy_static.frank_wolfe(X_t)
            ind_n = np.random.choice(X_t.shape[0], 1, p=lam_f)
            
            x_n = X_t[ind_n][0]
            y_n = x_n @ self.theta_star + self.sigma * np.random.randn()
            
            self.V += np.outer(x_n, x_n)
            S += x_n * y_n
            self.theta = np.linalg.inv(self.V) @ S
            
            if t%logging_period == 0:
                self.arm_sequence.append(np.argmax(self.X @ self.theta))

            