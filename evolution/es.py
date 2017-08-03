# dependencies
import numpy as np
import threading
from collections import deque
import tensorflow as tf


class EvolutionStrategy(object):

    def __init__(self, model_weights, reward_func, population_size, sigma, learning_rate, tensorboard):
        np.random.seed(0)
        self.weights = model_weights
        self.get_reward = reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate
        self.TENSORBOARD = tensorboard

    def get_model_weights(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA*i
            weights_try.append(w[index] + jittered)
        return weights_try
    
    def run(self, iterations, print_step=10):
        for iteration in range(iterations):

            if iteration % print_step == 0:
                print('iteration(%d) -> reward: %f' % (iteration, self.get_reward(self.weights)))

            population = []
            rewards = np.zeros(self.POPULATION_SIZE)
            for i in range(self.POPULATION_SIZE):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)

            for i in range(self.POPULATION_SIZE):
                weights_try = self.get_model_weights(self.weights, population[i])
                rewards[i]  = self.get_reward(weights_try)

            rewards = (rewards - np.mean(rewards)) / np.std(rewards)
                                 
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = w + self.LEARNING_RATE/(self.POPULATION_SIZE*self.SIGMA) * np.dot(A.T, rewards).T
    
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
            rewards[i]  = self.get_reward(weights_try)

        rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        return_queue.append([population, rewards])
 
    
    def run_dist(self, iterations, print_step=10, num_workers=1):
        for iteration in range(iterations//num_workers):
            
            if iteration % print_step == 0:
                print('iteration_dist(%d) -> reward: %f' % (iteration, self.get_reward(self.weights)))
                        
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
           
            
            
           
            
            

