#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Importing bits of the basic, multilayer perceptron model from here:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/multilayer_perceptron.py

n_hidden_1 = 256
n_hidden_2 = 128
n_input = 784
n_classes = 10

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

def init_variable(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

weights = { 
    'h1': init_variable([n_input, n_hidden_1]),
    'h2': init_variable([n_hidden_1, n_hidden_2]),
    'out': init_variable([n_hidden_2, n_classes])
}

biases = {
    'b1': init_variable([n_hidden_1]),
    'b2': init_variable([n_hidden_2]),
    'out': init_variable([n_classes])
}

# w_h = init_variable([784, 625]) # create symbolic variables
# w_o = init_variable([625, 10])

# def model(X, w_h, w_o):
#     h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
#     return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

def multi_tron(X):
    l_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    l_2 = tf.add(tf.matmul(l_1, weights['h2']), biases['b2'])
    out_l = tf.matmul(l_2, weights['out']) + biases['out']
    return out_l

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# py_x = model(X, w_h, w_o)
py_x = multi_tron(X)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x, labels=Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        if i % 10 == 0:
            print('Iteration:', i, '| Accuracy:', np.mean(np.argmax(teY, axis=1) == sess.run(predict_op, feed_dict={X: teX})))
            print('-'*30)
