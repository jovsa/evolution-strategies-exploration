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


#x_train = mnist.train.images
#x_valid = mnist.validation.images
#x_test = mnist.test.images


x_train = mnist.train.images.reshape(-1, img_rows, img_cols, 1)
x_valid = mnist.validation.images.reshape(-1, img_rows, img_cols, 1)
x_test = mnist.test.images.reshape(-1, img_rows, img_cols, 1)

y_train = mnist.train.labels
y_valid = mnist.validation.labels
y_test = mnist.test.labels

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                  activation='relu',
                  input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer= Adam(), loss='mse')



# # NN model definition
# input_layer = Input(shape=(784,))
# layer_1 = Dense(784)(input_layer)
# output_layer = Dense(num_classes, activation='softmax')(layer_1)
# model = Model(input_layer, output_layer)
# model.compile(Adam(), 'mse', metrics=['accuracy'])


def get_reward(weights):
    start_index = np.random.choice(y_train.shape[0]-batch_size-1,1)[0]
    solution = y_train[start_index:start_index+batch_size]
    inp = x_train[start_index:start_index+batch_size]

    model.set_weights(weights)
    prediction = model.predict(inp)

    reward = -np.sum(np.square(solution - prediction))
    return reward



prediction = model.predict(x_test)
print('test set accuracy - PRIOR:', np.mean(np.equal(np.argmax(prediction,1), np.argmax(y_test,1))))


prediction = model.predict(x_valid)
print('validation set accuracy - PRIOR:', np.mean(np.equal(np.argmax(prediction,1), np.argmax(y_valid,1))))


es = EvolutionStrategy(model.get_weights(), get_reward, population_size=50, sigma=0.1, learning_rate=0.001)
es.run(10, print_step=1)
#es.run_dist(300, print_step=1, num_workers=4)


prediction = model.predict(x_test)
print('test set accuracy - POST:', np.mean(np.equal(np.argmax(prediction,1), np.argmax(y_test,1))))

prediction = model.predict(x_valid)
print('validation set accuracy - POST:', np.mean(np.equal(np.argmax(prediction,1), np.argmax(y_valid,1))))

end_time = time.time()
print("Total Time usage: " + str(timedelta(seconds=int(round(end_time - start_time)))))

