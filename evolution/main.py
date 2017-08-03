# dependencies
import os
import numpy as np
import time
from datetime import timedelta
import pandas as pd
from es import EvolutionStrategy

from keras.models import Model, Input, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam # not important as there's no training here.
import tensorflow as tf
from keras import backend as K

start_time = time.time()

# data load
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# to run model evluation on 1 core
config = tf.ConfigProto(intra_op_parallelism_threads=1, 
                        inter_op_parallelism_threads=1, 
                        allow_soft_placement=True, device_count = {'CPU': 1, 'GPU':0})
session = tf.Session(config=config)
K.set_session(session)


# NN params
batch_size = 128
num_classes = 10
# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)


x_train = mnist.train.images
x_val = mnist.validation.images
x_test = mnist.test.images

y_train = mnist.train.labels
y_val = mnist.validation.labels
y_test = mnist.test.labels


# NN model definition
input_layer = Input(shape=(784,))
layer_1 = Dense(784)(input_layer)
output_layer = Dense(num_classes, activation='softmax')(layer_1)
model = Model(input_layer, output_layer)
model.compile(Adam(), 'mse', metrics=['accuracy'])


def get_reward(weights, calc_metrics = False):
    start_index = np.random.choice(y_train.shape[0]-batch_size-1,1)[0]
    solution = y_train[start_index:start_index+batch_size]
    inp = x_train[start_index:start_index+batch_size]

    model.set_weights(weights)
    prediction = model.predict(inp)
    
    metrics = {}
    if calc_metrics:
        metrics['accuracy_test'] = np.mean(np.equal(np.argmax(model.predict(x_test),1), np.argmax(y_test,1)))
        metrics['accuracy_val'] = np.mean(np.equal(np.argmax(model.predict(x_val),1), np.argmax(y_val,1)))
        metrics['accuracy_train'] = np.mean(np.equal(np.argmax(model.predict(inp),1), np.argmax(solution,1)))
       
    reward = -np.sum(np.square(solution - prediction))
    return reward, metrics


runs = {}
start_run = 1 # pick tot_runs > 0 if doing hyperparam search else 0
tot_runs = 100 # pick tot_runs > 1 if doing hyperparam search else 1
for i in range(start_run, tot_runs):
    npop = np.random.random_integers(1, 500, 1)[0]
    
    sample = np.random.rand(np.maximum(0,npop))
    sample_std = np.std(sample)
    sigma = np.round(np.sqrt(np.random.chisquare(sample_std,1)),2)[0]
    
    learning_rate_selection = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    alpha = np.random.choice(learning_rate_selection)
    
    hyperparams = [npop, sigma, alpha]
            
    chosen_before = False
    for key in runs.keys():
        if runs[key] == hyperparams:
            chosen_before = True
            
            print('skipping run, as hyperparams [{}] have been chosen before'.format(hyperparams))

    if not chosen_before:
        runs[i] = hyperparams        
        
        #default
        if i ==0:
            npop = 50
            sigma = 0.1
            alpha = 0.001
        print('hyperparam chosen npop:{}  sigma:{} alpha:{}'.format(npop, sigma, alpha))

        es = EvolutionStrategy(model.get_weights(), get_reward, population_size=npop, 
                               sigma=sigma, 
                               learning_rate=alpha)



        num_iterations = 1000 # for hyperparm search = 1000
        print_steps = 10 #for hyperparm search = 10
        num_workers = 1
        metrics = es.run(num_iterations, print_steps)
        #es.run_dist(num_iterations, print_steps, num_workers)

        print('saving results')
        results = pd.DataFrame(np.array(metrics).reshape(int((num_iterations//print_steps)), 6), 
                               columns=list(['run_name', 'iteration',
                                             'timestamp',
                                             'accuracy_test',
                                             'accuracy_val', 
                                             'accuracy_train']))
        RUN_SUMMARY_LOC = '../run_summaries/hyperparam_search'
        filename = os.path.join(RUN_SUMMARY_LOC, results['run_name'][0] + '.csv')
        print('writing results to disk')
        results.to_csv(filename, sep=',')
                
end_time = time.time()
print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))