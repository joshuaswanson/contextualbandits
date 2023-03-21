import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import multiprocessing as mp


class TopTwoAlgorithm(object):
    def __init__(self, X, theta_star, T, sigma=1, name=""):
        self.X = X
        self.d = X.shape[1]
        self.theta = np.zeros(self.d)
        self.theta_star = theta_star
        self.B = np.matmul(X.reshape(-1,self.d,1), X.reshape(-1,1,self.d))
        self.T = T
        self.V = np.eye(self.d)
        self.sigma = sigma
        self.name = name
        self.arm_sequence = []  # TODO: change this variable name
        
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

            min_idx = np.argmin((x_1 - x_2) @ np.linalg.inv(self.V + self.B) @ (x_1 - x_2))
            x_n = self.X[min_idx]
            y_n = self.theta_star @ x_n + self.sigma * np.random.randn()

            self.V += np.outer(x_n, x_n)
            S += x_n * y_n
            self.theta = np.linalg.inv(self.V) @ S
            
            errs.append(np.linalg.norm(self.theta - self.theta_star))
            self.arm_sequence.append(np.argmax(self.X @ self.theta))

            if t%logging_period == 0:
                print('toptwo run', self.name, 'iter', t,'\n')
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
            self.arm_sequence.append(np.argmax(self.X@theta))

            if t%logging_period == 0:
                print('ts run', self.name, 'iter', t,'\n')
                if verbose: 
                    plt.xlabel('iteration')
                    plt.ylabel(r'$\|\theta_*-\hat{\theta}\|$', rotation=0, labelpad=30)
                    plt.plot(errs);
                    plt.show()
                    clear_output(wait=True)
                    
                    
class XYStatic(object):
    
    def __init__(self, X, theta_star, T, sigma=1, name=""):
        self.X = X
        self.n, self.d = X.shape
        self.sigma = sigma
        self.arm_sequence = []
        self.V = np.eye(self.d)
        self.theta_star = theta_star
        self.name = name
        self.T = T
        self.Y = self.compute_Y(X)
        
        
    def run(self, logging_period=1):
        lam_f,_,_ = FW(self.X, self.Y)
        S = 0
        for t in range(self.T):
            idx = np.random.choice(self.n, p=lam_f)
            x_n = self.X[idx]
            y_n = x_n @ self.theta_star + self.sigma*np.random.randn()
            self.V += np.outer(x_n, x_n)
            S += x_n * y_n
            theta = np.linalg.inv(self.V) @ S 
            self.arm_sequence.append(np.argmax(self.X@theta))        
            if t%logging_period == 0:
                print('ts run', self.name, 'iter', t,'\n')
                
    
    @staticmethod
    def compute_Y(X):
        #TODO: change it to one-line
        res = []
        for i in range(X.shape[0]):
            for j in range(i+1, X.shape[1]):
                res.append(X[i] - X[j])
        return np.array(res)

class XYAdaptive(object):
    
    def __init__(self, X, theta_star, T, sigma=1, name=""):
        self.X = X
        self.d = X.shape[1]
        #self.theta = np.zeros(self.d)
        self.sigma = sigma
        self.arm_sequence = []
        self.V = np.eye(self.d)
        self.theta_star = theta_star
        self.name = name
        self.T = T
        self.k = 20
        self.Y = XYStatic.compute_Y(X)
        # self.k = k #TODO: add this later
        
        
        
    def run(self, logging_period=1):
        S = 0
        theta = np.zeros(self.d)
        for t in range(self.T):
            theta_mat = np.random.multivariate_normal(mean=theta, 
                                                      cov=self.V, size=self.k)
            max_x_vec = np.argmax(self.X @ theta_mat.transpose(), 
                                  axis=0)  # this should be dimension k
            
            X_t = self.X[max_x_vec]
            Y_t = XYStatic.compute_Y(X_t) #TODO: refactor this into utils.py
            
            lam_f, _, _ = FW(X_t, Y_t)
            
            ind_n = np.random.choice(X_t.shape[0], p=lam_f)
            
            x_n = X_t[ind_n]
            y_n = x_n @ self.theta_star + self.sigma * np.random.randn()
            
            self.V += np.outer(x_n, x_n)
            S += x_n * y_n
            theta = np.linalg.inv(self.V) @ S
            self.arm_sequence.append(np.argmax(self.X @ theta))
            
            if t%logging_period == 0:
                print('xy adaptive run', self.name, 'iter', t,'\n')
                #self.arm_sequence.append(np.argmax(self.X @ theta))


def FW(X, Y, reg_l2=0, iters=10000, 
       step_size=1, viz_step = 10000, 
       initial=None):
    n, d = X.shape
    I = np.eye(n)
    if initial is not None:
        design = initial
    else:
        design = np.ones(n)
        design /= design.sum()  
    eta = step_size
    grad_norms = []
    history = []
    
    for count in range(1, iters):
        A_inv = np.linalg.pinv(X.T@np.diag(design)@X + reg_l2*np.eye(d))        
        rho = np.diag(Y@A_inv@Y.T)
        y_opt = Y[np.argmax(rho),:]
        g = y_opt @ A_inv @ X.T
        g = -g * g
        
        eta = step_size/(count+2)
        imin = np.argmin(g)
        design = (1-eta)*design+eta*I[imin]
        grad_norms.append(np.linalg.norm(g - np.sum(g)/n*np.ones(n)))
        if count % (viz_step) == 0:
            history.append(np.max(rho))
            fig, ax = plt.subplots(1,2)
            ax[0].plot(grad_norms)
            ax[1].plot(design)
            
            plt.show()
    return design, rho, history
