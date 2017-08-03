# dependencies
import numpy as np
import time
from datetime import timedelta
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



chosen_population_size = 50
chosen_sigma = 0.1
chosen_learning_rate = 0.001
tensorboard_summaries = '../tensorboard_summaries/'
es = EvolutionStrategy(model.get_weights(), get_reward, population_size=chosen_population_size, 
                       sigma=chosen_sigma, 
                       learning_rate=chosen_learning_rate,
                       tensorboard_loc = tensorboard_summaries)
es.run(300, print_step=50)
#es.run_dist(100, print_step=10, num_workers=4)


end_time = time.time()
print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))

