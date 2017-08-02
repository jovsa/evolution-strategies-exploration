# dependencies
import numpy as np

from keras.models import Model, Input, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam # not important as there's no training here.
import tensorflow as tf
from keras import backend as K

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


config = tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=2, allow_soft_placement=True, device_count = {'CPU': 1})
session = tf.Session(config=config)
K.set_session(session)


batch_size = 128
num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)

x_train = mnist.train.images
x_valid = mnist.validation.images
x_test = mnist.test.images
print('x_train:', np.shape(x_train))
print('x_valid:', np.shape(x_valid))
print('x_test:', np.shape(x_test))

y_train = mnist.train.labels
y_valid = mnist.validation.labels
y_test = mnist.test.labels
print('y_train:', np.shape(y_train))
print('y_valid:', np.shape(y_valid))
print('y_test:', np.shape(y_test))

input_layer = Input(shape=(784,))
layer = Dense(1500)(input_layer)
output_layer = Dense(num_classes, activation='softmax')(layer)
model = Model(input_layer, output_layer)
model.compile(Adam(), 'mse', metrics=['accuracy'])


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



def get_reward(weights):
    start_index = np.random.choice(y_train.shape[0]-batch_size-1,1)[0]
    solution = y_train[start_index:start_index+batch_size]
    inp = x_train[start_index:start_index+batch_size]

    model.set_weights(weights)
    prediction = model.predict(inp)

    reward = -np.sum(np.square(solution - prediction))
    return reward


prediction = model.predict(x_test)
solution = y_test
print('test set accuracy - PRIOR:', np.mean(np.equal(np.argmax(prediction,1), np.argmax(solution,1))))


prediction = model.predict(x_valid)
solution = y_valid
print('validation set accuracy - PRIOR:', np.mean(np.equal(np.argmax(prediction,1), np.argmax(solution,1))))




es = EvolutionStrategy(model.get_weights(), get_reward, population_size=50, sigma=0.1, learning_rate=0.01)
es.run(20, print_step=1)


prediction = model.predict(x_test)
solution = y_test
print('test set accuracy - POST:', np.mean(np.equal(np.argmax(prediction,1), np.argmax(solution,1))))


prediction = model.predict(x_valid)
solution = y_valid
print('validation set accuracy - POST:', np.mean(np.equal(np.argmax(prediction,1), np.argmax(solution,1))))
