import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import multiprocessing as mp
from bandit_type import Linear, Concept
from utils import *
from distribution import *

class GeneralThompson(Concept):
    def __init__(self, X, gen_star, T, sigma, name):
        super().__init__(X, gen_star, T, sigma, name)       
        if type(gen_star) is not GenericFunction:
            self.gen_star = GenericFunction(gen_star, sigma)
        self.pi = Kitten([], [], None, sigma)
        self.B = 20
        self.pulled = []
        self.name = name
        self.delta=.0001
        
    def run(self, logging_period=1, k=10):
        for t in range(self.T):
            f1 = self.pi.sample()
            best_idx = np.argmax(f1.evaluate(self.X))
            
            x_n = self.X[best_idx]
            y_n = self.gen_star.pull(x_n)
            print(x_n, y_n)
            self.pi.update_posterior(x_n, y_n)
            
            self.fhat = self.pi.map()
            idx = np.argmax(self.fhat.evaluate(self.X))

            self.pulled.append(best_idx)
            self.arms_recommended.append(idx)

            if t%logging_period == 0:
                print('general run', self.name, 'iter', t, "/", self.T, 'idx', best_idx, end="\r")
        quit = len(self.arms_recommended)
        rec = self.arms_recommended[-1]
        if quit < self.T:
            for i in range(quit, self.T):
                self.arms_recommended.append(rec)


# class GeneralTopTwo(Hypothesis):
#     def __init__(self, X, gen_star, T, sigma, name):
#         super().__init__(X, gen_star, T, sigma, name)       
#         self.pi = Kitten(f_star)
#         self.B = 20
#         self.pulled = []
#         self.name = name
#         self.delta=.0001
        
#     def run(self, logging_period=1, k=10):
#         for t in range(self.T):

#             f1 = self.pi.sample()
#             idx1 = np.argmax(f1.evaluate(self.X))
#             x1 = self.X[idx1]

#             a = 0
#             idx2 = idx1
#             while idx1 == idx2:
#                 f2s = self.pi.sample(k)  # TODO: sample 10 at a time
#                 for f2 in f2s:
#                     idx2 = np.argmax(f2.evaluate(self.X))
#                     if idx1 != idx2:
#                         break
#                 a+=k
#                 if a > 1/self.delta:
#                     break
#             if a > 1/self.delta:
#                 break
                    
#             x2 = self.X[idx2]

#             v = []               
#             for idx in range(self.n):
#                 x = self.X[idx]
#                 expected_diff = 0
#                 expected_diff_squared = 0
                
#                 for b1 in range(self.B):
#                     gen_b1 = self.pi.sample()
#                     y_b1 = gen_b1.pull(x)
#                     weight = t # we should change this weighting to simulating t observations from x
#                     pi_plus = self.pi.update_posterior(x*weight, y_b1, copy=True)
#                     gen_b2 = pi_plus.sample()
#                     expected_diff += ( gen_b2.evaluate(x1) - gen_b2.evaluate(x2) ) 
#                     expected_diff_squared += ( gen_b2.evaluate(x1) - gen_b2.evaluate(x2) )**2
#                 v.append( expected_diff_squared/self.B - (expected_diff/self.B)**2 )
                
#             min_idx = np.argmin(v)
#             self.pulled.append(min_idx)
#             x_n = self.X[min_idx]
#             y_n = self.gen_star.pull(x_n)
#             self.pi.update_posterior(x_n, y_n)
            
#             fhat = self.pi.map()
#             idx = np.argmax(fhat.evaluate(self.X))
#             self.arms_recommended.append(idx)

#             if t%logging_period == 0:
#                 print('general run', self.name, 'iter', t, "/", self.T, end="\r")
#         quit = len(self.arms_recommended)
#         rec = self.arms_recommended[-1]
#         if quit < self.T:
#             for i in range(quit, self.T):
#                 self.arms_recommended.append(rec)

