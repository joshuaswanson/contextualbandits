import sys
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import multiprocessing as mp

import library_concept
import library_linear
import instance

import utils
import json


    

def worker(alg, X, Y, f_star, T, sigma, ix):
    print('algorithm', alg, 'name', alg['name'])
    np.random.seed()
    algorithm_instance = utils.get_alg(alg, X, Y, f_star, T, sigma, ix)
    algorithm_instance.run(logging_period=100)
    return algorithm_instance.arms_recommended

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--T', type=int)
    #parser.add_argument('--reps', type=int)
    #parser.add_argument('--cpu', default=10, type=int)
    #parser.add_argument('--parallelize', default='mp', type=str)
    parser.add_argument('--config', type=str)
    parser.add_argument('--path', default=os.getcwd(), type=str)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        params = json.load(f)
        print(params)


    T = params['global']['T']  # time steps
    sigma = params['global']['sigma']
    reps = params['global']['reps']  # repetitions of the algorithm
    cpu = params['global']['cpu']
    path = args.path
    print('OUR PATH', path)

    X, f_star = instance.get_instance(**params['global']['instance'])
    Y = utils.compute_Y(X)
    
    algorithms = [alg for alg in params['algs'] if alg['active']]
    runs = []
    
    for alg in algorithms:
        runs += [(alg, X, Y, f_star, T, sigma, i) for i in range(reps)]

    if params['global']['parallelize'] == 'mp':
        pool = mp.Pool(cpu, maxtasksperchild=1000)
        all_results = pool.starmap(worker, runs)
    else:
        import ray
        ray.init(address='128.208.6.83:6379',)
        #worker = ray.remote(scheduling_strategy="SPREAD")(worker)
        #all_results = ray.get([worker.remote(*a) for a in runs])
        all_results = []
        for a in runs:
            all_results.append(worker(*a))

    
    idx_star = np.argmax(f_star.evaluate(X))  # index of best arm
    K = X.shape[0]
    d = X.shape[1]
    xaxis = np.arange(T)
    for i,alg in enumerate(algorithms):
        results = all_results[reps*i: reps*(i+1)]
        print('results', [len(results[i]) for i in range(reps)])
        m = (results == idx_star).mean(axis=0)
        s = (results == idx_star).std(axis=0)/np.sqrt(reps)
        plt.plot(xaxis, m)
        plt.fill_between(xaxis, m - s, m + s, alpha=0.2, label=alg['alg_class'])

    
    plt.xlabel('time')
    plt.ylabel('identification rate')
    #plt.title(f'K {K} d {d}')
    plt.legend()
    plt.savefig(path+'/results.png')