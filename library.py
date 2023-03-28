import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import multiprocessing as mp
from bandit_type import Linear
from utils import *
import distribution

class ThompsonSampling(Linear):

    def __init__(self, X, Y, theta_star, T, sigma, name):
        super().__init__(X, Y, theta_star, T, sigma, name)
        self.B = np.matmul(X.reshape(-1,self.d,1), X.reshape(-1,1,self.d))
        self.Vinv = np.linalg.inv(self.V)

    def run(self, logging_period=1):
        theta = np.zeros(self.d)
        S = 0
        for t in range(self.T):
            theta_hat = np.random.multivariate_normal(theta, self.Vinv)
            best_idx = np.argmax(self.X @ theta_hat)
            x_n = self.X[best_idx]
            y_n = x_n @ self.theta_star + self.sigma*np.random.randn()
            self.V += np.outer(x_n, x_n)
            self.Vinv = fast_rank_one(self.Vinv, x_n)
            S += x_n * y_n
            theta = np.linalg.inv(self.V) @ S
            self.arms_recommended.append(np.argmax(self.X @ theta))
            if t%logging_period == 0:
                print('ts run', self.name, 'iter', t, "/", self.T, end="\r")


class TopTwoAlgorithm(Linear):
        
    def __init__(self, X, Y, theta_star, T, sigma, name):
        super().__init__(X, Y, theta_star, T, sigma, name)
        self.B = np.matmul(X.reshape(-1,self.d,1), X.reshape(-1,1,self.d))
        self.Vinv = np.linalg.inv(self.V)
        self.toptwo = []
        self.pulled = []
        self.k = 10
        
    def run(self, logging_period=1):
        theta = np.zeros(self.d)
        S = 0
        
        for t in range(self.T):
            theta_1 = np.random.multivariate_normal(theta, self.Vinv)
            best_idx = np.argmax(self.X @ theta_1)
            x_1 = self.X[best_idx]

            # try it for a first time
            theta_2 = np.random.multivariate_normal(theta, self.Vinv)
            best_idx_2 = np.argmax(self.X@theta_2)

            while best_idx == best_idx_2:
                # draw k theta's and compute the best x at the same time to make it faster
                theta_2_mat = np.random.multivariate_normal(mean=theta, 
                                                      cov=self.Vinv, size=self.k)
                max_x2_vec = np.argmax(self.X @ theta_2_mat.transpose(), axis=0)
                #print(max_x2_vec!=best_idx)
                if any(max_x2_vec!=best_idx): # if there is some index that is different
                    # find the first place where they are different
                    #print(np.where(max_x2_vec != best_idx)[0])
                    best_idx_2 = max_x2_vec[np.where(max_x2_vec != best_idx)[0][0]]
            
            x_2 = self.X[best_idx_2]
            self.toptwo.append([best_idx, best_idx_2])
            

            min_idx = np.argmin((x_1 - x_2) @ np.linalg.inv(self.V + self.B) @ (x_1 - x_2))
            self.pulled.append(min_idx)
            x_n = self.X[min_idx]
            y_n = self.theta_star @ x_n + self.sigma * np.random.randn()

            self.V += np.outer(x_n, x_n)
            self.Vinv = fast_rank_one(self.Vinv, x_n)
            S += x_n * y_n
            theta = self.Vinv @ S
            self.theta = theta
            self.arms_recommended.append(np.argmax(self.X @ theta))

            if t%logging_period == 0:
                print('toptwo run', self.name, 'iter', t, "/", self.T, end="\r")

                    
class XYStatic(Linear):
    
    def __init__(self, X, Y, theta_star, T, sigma, name):
        super().__init__(X, Y, theta_star, T, sigma, name)
        #self.Y = compute_Y(X)
        
        
    def run(self, logging_period=1):
        lam_f,_,_ = FW(self.X, self.Y, iters=1000)
        del self.Y
        S = 0
        for t in range(self.T):
            idx = np.random.choice(self.n, p=lam_f)
            x_n = self.X[idx]
            y_n = x_n @ self.theta_star + self.sigma*np.random.randn()
            self.V += np.outer(x_n, x_n)
            S += x_n * y_n
            theta = np.linalg.inv(self.V) @ S 
            self.arms_recommended.append(np.argmax(self.X @ theta))        
            if t%logging_period == 0:
                print('xy static run', self.name, 'iter', t, "/", self.T, end="\r")

    
class XYAdaptive(Linear):
    
    def __init__(self, X, Y, theta_star, T, sigma, name):
        super().__init__(X, Y, theta_star, T, sigma, name)
        self.k = 5
        self.Vinv = np.linalg.inv(self.V)
        #self.Y = compute_Y(X)
        # self.k = k #TODO: add this later

    def run(self, logging_period=1, FW_verbose=False, FW_logging_period=100):
        S = 0
        lam_f = np.ones(self.n)/self.n
        theta = np.zeros(self.d)
        for t in range(self.T):
            theta_mat = np.random.multivariate_normal(mean=theta, 
                                                      cov=self.Vinv, size=self.k)
            max_x_vec = np.argmax(self.X @ theta_mat.transpose(), axis=0)  
            # this should be dimension k
            
            X_t = self.X[max_x_vec]
            Y_t = compute_Y(X_t)
            lam_f, _, _ = FW(self.X, Y_t, initial=lam_f, 
                             iters=20, step_size=1, 
                             logging_step=FW_logging_period, verbose=FW_verbose)
            
            ind_n = np.random.choice(self.X.shape[0], p=lam_f)
            
            x_n = self.X[ind_n]
            y_n = x_n @ self.theta_star + self.sigma * np.random.randn()
            
            self.V += np.outer(x_n, x_n)
            self.Vinv = np.linalg.inv(self.V)
            S += x_n * y_n
            theta = self.Vinv @ S
            self.arms_recommended.append(np.argmax(self.X @ theta))
            
            if t%logging_period == 0:
                print('xy adaptive run', self.name, 'iter', t, "/", self.T, end="\r")
