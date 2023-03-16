import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import multiprocessing as mp
import library
from importlib import reload
import argparse

reload(library)

def worker(algorithm, X, theta_star, T, sigma, name):
    np.random.seed()
    algorithm_class = algorithm(X, theta_star, T, sigma, name)
    algorithm_class.run()
    return algorithm_class.best_x


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', type=int)
    parser.add_argument('--reps', type=int)
    args = parser.parse_args()
    
    
    T = args.T  # time steps
    reps = args.reps  # repetitions of the algorithm
    d = 10  # dimension of arms
    K = 100  # number of arms
    X = np.random.randn(K, d)  # arms
    theta_star = np.random.randn(d)  # 
    idx_star = np.argmax(X@theta_star)  # index of best arm
    
    pool = mp.Pool(40)
    args1 = [(library.TopTwoAlgorithm, X, theta_star, T, 1, i) for i in range(reps)]
    args2 = [(library.ThompsonSampling, X, theta_star, T, 1, i) for i in range(reps)]
    
    results1 = pool.starmap(worker, args1)
    results2 = pool.starmap(worker, args2)
    
    m1 = (results1 == idx_star).mean(axis=0)
    m2 = (results2 == idx_star).mean(axis=0)
    
    s1 = (results1 == idx_star).std(axis=0)/np.sqrt(reps)
    s2 = (results2 == idx_star).std(axis=0)/np.sqrt(reps)
    
    xaxis = np.arange(T)
    
    np.random.seed()
    algorithm_class = library.TopTwoAlgorithm(X, theta_star, T, 1, 10)
    algorithm_class.run()

    plt.plot(xaxis, m1)
    plt.fill_between(xaxis, m1 - 1.96 * s1, m1 + 1.96 * s1,
                     color='blue', alpha=0.2, label='Top Two Posterior')
    
    plt.plot(xaxis, m2)
    plt.fill_between(xaxis, m2 - 1.96 * s2, m2 + 1.96 * s2,
                     color='orange', alpha=0.2, label='Thompson Sampling')
    
    plt.xlabel('time step')
    plt.ylabel('probability optimal arm is selected')
    
    plt.legend()
    plt.savefig('results.png')