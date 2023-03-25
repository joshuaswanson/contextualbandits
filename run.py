import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import multiprocessing as mp
import library
from importlib import reload
import argparse
from settings import *
import utils
import ray

reload(library)

def get_alg(name):
    if name == 'ThompsonSampling':
        return library.ThompsonSampling
    elif name == 'TopTwoAlgorithm':
        return library.TopTwoAlgorithm
    elif name == 'XYStatic':
        return library.XYStatic
    else:
        return library.XYAdaptive
    
#@ray.remote(scheduling_strategy="SPREAD")
def worker(algorithm, X, Y, theta_star, T, sigma, name):
    np.random.seed()
    algorithm = get_alg(algorithm)
    #Y = np.frombuffer(Y_buffer).reshape((X.shape[0]*(X.shape[0]-1)//2,X.shape[1]))
    algorithm_instance = algorithm(X, Y, theta_star, T, sigma, name)
    algorithm_instance.run(logging_period=1)
    return algorithm_instance.arms_chosen


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int)
    parser.add_argument('--reps', type=int)
    parser.add_argument('--cpu', default=10, type=int)
    parser.add_argument('--parallelize', default='mp', type=str)
    args = parser.parse_args()
    
    
    T = args.T  # time steps
    reps = args.reps  # repetitions of the algorithm
    cpu = args.cpu
    
    
    d = 20
    X,theta_star = soare(d, alpha=1/np.sqrt(d))
    K = X.shape[0]
    idx_star = np.argmax(X @ theta_star)  # index of best arm
    Y = utils.compute_Y(X)
    
    
    algorithms = [
        'ThompsonSampling',
        'TopTwoAlgorithm',
        'XYStatic',
        'XYAdaptive'
    ]
     
    
    runs = []
    for algorithm in algorithms:
        runs += [(algorithm, X, Y, theta_star, T, 1, i) for i in range(reps)]

    #if args.parallelize == 'mp':
    pool = mp.Pool(cpu, maxtasksperchild=1000)
    all_results = pool.starmap(worker, runs)
#     else:
#         ray.init(address='128.208.6.83:6379')
#         all_results = ray.get([worker.remote(**a) for a in runs])
    
    xaxis = np.arange(T)
    for i,algorithm in enumerate(algorithms):
        results = np.array(all_results[reps*i: reps*(i+1)])
        m = (results == idx_star).mean(axis=0)
        s = (results == idx_star).std(axis=0)/np.sqrt(reps)
        plt.plot(xaxis, m)
        plt.fill_between(xaxis, m - 1.96 * s, 
                         m + 1.96 * s, alpha=0.2, label=algorithm)

    
    plt.xlabel('time')
    plt.ylabel('identification rate')
    plt.title(f'K {K} d {d}')
    plt.legend()
    plt.savefig('results.png')