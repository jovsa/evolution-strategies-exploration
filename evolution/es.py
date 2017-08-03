# dependencies
import numpy as np
import threading
from collections import deque


class EvolutionStrategy(object):

    def __init__(self, weights, get_reward_func, population_size=50, sigma=0.1, learning_rate=0.001):
        np.random.seed(0)
        self.weights = weights
        self.get_reward = get_reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.LEARNING_RATE = learning_rate


    def _get_weights_try(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA*i
            weights_try.append(w[index] + jittered)
        return weights_try


    def get_weights(self):
        return self.weights
    
    def run(self, iterations, print_step=10):
        for iteration in range(iterations):

            if iteration % print_step == 0:
                print('iter %d. reward: %f' % (iteration, self.get_reward(self.weights)))

            population = []
            rewards = np.zeros(self.POPULATION_SIZE)
            for i in range(self.POPULATION_SIZE):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)

            for i in range(self.POPULATION_SIZE):
                weights_try = self._get_weights_try(self.weights, population[i])
                rewards[i]  = self.get_reward(weights_try)

            rewards = (rewards - np.mean(rewards)) / np.std(rewards)
                                 
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = w + self.LEARNING_RATE/(self.POPULATION_SIZE*self.SIGMA) * np.dot(A.T, rewards).T
    
    def worker(self, worker_name, return_queue):
        print('worker:',  worker_name)
        population = []     
        rewards = np.zeros(self.POPULATION_SIZE)
            
        for i in range(self.POPULATION_SIZE):
            x = []
            for w in self.weights:
                x.append(np.random.randn(*w.shape))
            population.append(x)
                  
        for i in range(self.POPULATION_SIZE):
            weights_try = self._get_weights_try(self.weights, population[i])
            rewards[i]  = self.get_reward(weights_try)

        rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        return_queue.append([population, rewards])
 
    
    def run_dist(self, iterations, print_step=10, num_workers=1):
        for iteration in range(iterations):
            
            if iteration % print_step == 0:
                print('iter %d. reward: %f' % (iteration, self.get_reward(self.weights)))
            
            
            return_queue = deque()
            
            #for worker in num_workers:
            t1= threading.Thread(target=self.worker, args=(str(1), return_queue))
            t2= threading.Thread(target=self.worker, args=(str(2), return_queue))
            t3= threading.Thread(target=self.worker, args=(str(3), return_queue))
            
            t1.start()
            t2.start()
            t3.start()
            
            t1.join()
            t2.join()
            t3.join()
            
            #print('work done')
            for worker_output in return_queue:
                population = worker_output[0]
                rewards = worker_output[1]
                
                for index, w in enumerate(self.weights):
                    A = np.array([p[index] for p in population])
                    self.weights[index] = w + self.LEARNING_RATE/(self.POPULATION_SIZE*self.SIGMA) * np.dot(A.T, rewards).T
            
            #print('population', len(population))
            #print('rewards', len(rewards))
            
            #rewards = (rewards - np.mean(rewards)) / np.std(rewards)
            
            
           
            
            

