# dependencies
import numpy as np
import threading
import time
from collections import deque

# main class for implimenting Evolution Strategies
class EvolutionStrategy(object):

    def __init__(self, model_weights, reward_func, population_size, sigma, learning_rate):
        np.random.seed(0)
        self.weights = model_weights
        self.get_reward = reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate
       
    def get_model_weights(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA*i
            weights_try.append(w[index] + jittered)
        return weights_try
    
    # implimention of Algorithm 1: Evolution Strategies by Salimans et al., OpenAI [1], p.2/12
    def run(self, iterations, print_step=10):
        metrics = []
        run_name = ('npop={0:}_sigma={1:}_alpha={2:}_iters={3:}_type={4:}').format(self.POPULATION_SIZE ,
                                                                                   self.SIGMA ,
                                                                                   self.LEARNING_RATE,
                                                                                   iterations,
                                                                                   'run')
        
        for iteration in range(iterations):
            
            # checking fitness
            if iteration % print_step == 0:
                _, return_metrics = self.get_reward(self.weights, calc_metrics=True)                  
                print('iteration({}) -> reward: {}'.format(iteration, return_metrics))
                metrics.append([run_name, iteration, time.time(),
                                return_metrics['accuracy_test'], 
                                return_metrics['accuracy_val'], 
                                return_metrics['accuracy_train']])

            population = []
            rewards = np.zeros(self.POPULATION_SIZE)
            for i in range(self.POPULATION_SIZE):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)

            for i in range(self.POPULATION_SIZE):
                weights_try = self.get_model_weights(self.weights, population[i])
                rewards[i], _  = self.get_reward(weights_try)

            rewards = (rewards - np.mean(rewards)) / np.std(rewards)
                                 
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = w + self.LEARNING_RATE/(self.POPULATION_SIZE*self.SIGMA) * np.dot(A.T, rewards).T
        return metrics
    
    def worker(self, worker_name, return_queue):
        population = []     
        rewards = np.zeros(self.POPULATION_SIZE)
            
        for i in range(self.POPULATION_SIZE):
            x = []
            for w in self.weights:
                x.append(np.random.randn(*w.shape))
            population.append(x)
                  
        for i in range(self.POPULATION_SIZE):
            weights_try = self.get_model_weights(self.weights, population[i])
            rewards[i], _  = self.get_reward(weights_try)

        rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        return_queue.append([population, rewards])
 
    # Algorithm 2: Parallelized Evolution Strategies by Salimans et al., OpenAI [1], p.3/12
    def run_dist(self, iterations, print_step=10, num_workers=1):
        metrics = []
        run_name = ('npop={0:}_sigma={1:}_alpha={2:}_iters={3:}_type={4:}').format(self.POPULATION_SIZE ,
                                                                                   self.SIGMA ,
                                                                                   self.LEARNING_RATE,
                                                                                   iterations,
                                                                                   'run_dist')
        for iteration in range(iterations//num_workers):
            
            # checking fitness
            if iteration % print_step == 0:
                _, return_metrics = self.get_reward(self.weights, calc_metrics=True)                  
                print('iteration({}) -> reward: {}'.format(iteration, return_metrics))
                metrics.append([run_name, iteration, time.time(),
                                return_metrics['accuracy_test'], 
                                return_metrics['accuracy_val'], 
                                return_metrics['accuracy_train']])
                        
            return_queue = deque()
            jobs = []
            
            for worker in range(0, num_workers):
                # picking custom seed for each worker
                np.random.seed(num_workers * 10) 
                job= threading.Thread(target=self.worker, args=(str(worker), return_queue))
                jobs.append(job)
                job.start()
                
            for job in jobs:
                job.join()
                
            population = []
            rewards = []
            
            for worker_output in return_queue:
                population.extend(worker_output[0])
                rewards.extend(worker_output[1])
                
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = w + self.LEARNING_RATE/(self.POPULATION_SIZE*self.SIGMA) * np.dot(A.T, rewards).T
        
           
            
            
           
            
            

