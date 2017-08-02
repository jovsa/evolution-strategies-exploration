import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../../MNIST_data/", one_hot=True)
chosen_image = np.random.randint(1,500)


def test_data_size_test():
    assert(np.shape(mnist.train.images)[0] == 
           np.shape(mnist.train.labels)[0]== 55000)

def test_data_size_train():
    assert(np.shape(mnist.test.images)[0] == 
           np.shape(mnist.test.labels)[0]== 10000)

def test_data_size_validation():
    assert(np.shape(mnist.validation.images)[0] == 
           np.shape(mnist.validation.labels)[0]== 5000)

def test_image_shape():
    assert(np.shape(mnist.train.images[chosen_image]) ==
    np.shape(mnist.test.images[chosen_image]) ==
    np.shape(mnist.validation.images[chosen_image]) == (784,))

def test_label_shape():
    assert(np.shape(mnist.train.labels[chosen_image]) ==
    np.shape(mnist.test.labels[chosen_image]) ==
    np.shape(mnist.validation.labels[chosen_image]) == (10,))
