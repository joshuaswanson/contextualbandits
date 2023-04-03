import sys
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import pickle

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import multiprocessing as mp

import contextualbandits.library_concept as library_concept
import argparse
from instance import *
import utils


    

def worker(algorithm, X, Y, theta_star, T, sigma, name):
    print('algorithm', algorithm, 'name', name)
    def get_alg(name):
        if name == 'ThompsonSampling':
            return library_concept.ThompsonSampling
        elif name == 'TopTwoAlgorithm':
            return library_concept.TopTwoAlgorithm
        elif name == 'XYStatic':
            return library_concept.XYStatic
        elif name == 'General':
            return library_concept.General
        else:
            return library_concept.XYAdaptive
    np.random.seed()
    algorithm = get_alg(algorithm)
    #Y = np.frombuffer(Y_buffer).reshape((X.shape[0]*(X.shape[0]-1)//2,X.shape[1]))
    algorithm_instance = algorithm(X, Y, theta_star, T, sigma, name)
    algorithm_instance.run(logging_period=100)
    return algorithm_instance.arms_recommended

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int)
    parser.add_argument('--reps', type=int)
    parser.add_argument('--cpu', default=10, type=int)
    parser.add_argument('--parallelize', default='mp', type=str)
    parser.add_argument('--path', default=os.getcwd(), type=str)
    args = parser.parse_args()
    

    T = args.T  # time steps
    reps = args.reps  # repetitions of the algorithm
    cpu = args.cpu
    path = args.path
    print('OUR PATH', path)

    d = 20
    X,theta_star = soare(d, alpha=.1)
    K = X.shape[0]
    idx_star = np.argmax(X @ theta_star)  # index of best arm
    Y = utils.compute_Y(X)
    
    
    algorithms = [
        'ThompsonSampling',
        'TopTwoAlgorithm',
        'XYStatic',
        'General'
        #'XYAdaptive'
    ]
     
    
    runs = []
    for algorithm in algorithms:
        runs += [(algorithm, X, Y, theta_star, T, 1, i) for i in range(reps)]

    if args.parallelize == 'mp':
        pool = mp.Pool(cpu, maxtasksperchild=1000)
        all_results = pool.starmap(worker, runs)
    else:
        import ray
        ray.init(address='128.208.6.83:6379',)
        worker = ray.remote(scheduling_strategy="SPREAD")(worker)
        all_results = ray.get([worker.remote(*a) for a in runs])

    
    # file = open('/home/lalitj/contextualbandits/results.pkl', 'wb')
    # print(file)
    # pickle.dump(all_results, file)
    # file.close()
    # print(os.listdir('.'), os.getcwd())
    
    xaxis = np.arange(T)
    for i,algorithm in enumerate(algorithms):
        results = all_results[reps*i: reps*(i+1)]
        print([len(results[i]) for i in range(reps)])
        m = (results == idx_star).mean(axis=0)
        s = (results == idx_star).std(axis=0)/np.sqrt(reps)
        plt.plot(xaxis, m)
        plt.fill_between(xaxis, m - s, m + s, alpha=0.2, label=algorithm)

    
    plt.xlabel('time')
    plt.ylabel('identification rate')
    plt.title(f'K {K} d {d}')
    plt.legend()
    plt.savefig(path+'/results.png')