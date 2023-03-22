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

def worker(algorithm, X, theta_star, T, sigma, name):
    np.random.seed()
    algorithm = get_alg(algorithm)
    algorithm_instance = algorithm(X, theta_star, T, sigma, name)
    algorithm_instance.run(logging_period=1)
    print('run finished', algorithm, name)
    return algorithm_instance.arms_chosen


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int)
    parser.add_argument('--reps', type=int)
    parser.add_argument('--cpu', default=10, type=int)
    args = parser.parse_args()
    
    
    T = args.T  # time steps
    reps = args.reps  # repetitions of the algorithm
    cpu = args.cpu
    
    K = 500
    d = 6
    X,theta_star = sphere(K, d)
    idx_star = np.argmax(X @ theta_star)  # index of best arm
    
    algorithms = [
        'ThompsonSampling',
        'TopTwoAlgorithm',
        #'XYStatic'
        #library.XYAdaptive
    ] 
     
    xaxis = np.arange(T)
    pool = mp.Pool(cpu, maxtasksperchild=1000)
    runs = []
    for algorithm in algorithms:
        runs += [(algorithm, X, theta_star, T, 1, i) for i in range(reps)]
    print('num runs', len(runs))
    all_results = pool.starmap(worker, runs)

    for i,algorithm in enumerate(algorithms):
        results = np.array(all_results[reps*i: reps*(i+1)])
        m = (results == idx_star).mean(axis=0)
        s = (results == idx_star).std(axis=0)/np.sqrt(reps)
        plt.plot(xaxis, m)
        plt.fill_between(xaxis, m - 1.96 * s, m + 1.96 * s, alpha=0.2, label=algorithm)

#     for algorithm in algorithms:
#         pool = mp.Pool(cpu)
#         args = [(algorithm, X, theta_star, T, 1, i) for i in range(reps)]
#         results = pool.starmap(worker, args)
#         pool.close()
#         m = (results == idx_star).mean(axis=0)
#         s = (results == idx_star).std(axis=0)/np.sqrt(reps)
#         plt.plot(xaxis, m)
#         plt.fill_between(xaxis, m - 1.96 * s, m + 1.96 * s, alpha=0.2, label=algorithm.__name__)
    
#     args1 = [(library.TopTwoAlgorithm, X, theta_star, T, 1, i) for i in range(reps)]
#     args2 = [(library.ThompsonSampling, X, theta_star, T, 1, i) for i in range(reps)]
#     args3 = [(library.XYStatic, X, theta_star, T, 1, i) for i in range(reps)]
#     args4 = [(library.XYAdaptive, X, theta_star, T, 1, i) for i in range(reps)]
    
#     results1 = pool.starmap(worker, args1)
#     results2 = pool.starmap(worker, args2)
#     results3 = pool.starmap(worker, args3)
#     results4 = pool.starmap(worker, args4)
    
#     m1 = (results1 == idx_star).mean(axis=0)
#     m2 = (results2 == idx_star).mean(axis=0)
#     m3 = (results3 == idx_star).mean(axis=0)
#     m4 = (results4 == idx_star).mean(axis=0)
    
#     s1 = (results1 == idx_star).std(axis=0)/np.sqrt(reps)
#     s2 = (results2 == idx_star).std(axis=0)/np.sqrt(reps)
#     s3 = (results3 == idx_star).std(axis=0)/np.sqrt(reps)
#     s4 = (results4 == idx_star).std(axis=0)/np.sqrt(reps)
    
#     xaxis = np.arange(len(m1))

#     plt.plot(xaxis, m1)
#     plt.fill_between(xaxis, m1 - 1.96 * s1, m1 + 1.96 * s1,
#                      color='blue', alpha=0.2, label='Top two posterior')
    
#     plt.plot(xaxis, m2)
#     plt.fill_between(xaxis, m2 - 1.96 * s2, m2 + 1.96 * s2,
#                      color='orange', alpha=0.2, label='Thompson sampling')
    
#     plt.plot(xaxis, m3)
#     plt.fill_between(xaxis, m3 - 1.96 * s3, m3 + 1.96 * s3,
#                      color='green', alpha=0.2, label='XY static')
    
#     plt.plot(xaxis, m4)
#     plt.fill_between(xaxis, m4 - 1.96 * s4, m4 + 1.96 * s4,
#                      color='red', alpha=0.2, label='XY adaptive')
    
    plt.xlabel('time')
    plt.ylabel('identification rate')
    plt.title(f'K {K} d {d}')
    plt.legend()
    plt.savefig('results.png')