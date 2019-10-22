#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    return tf.matmul(X, w) # notice we use the same model as linear regression, this is because there is a baked in cost function which performs softmax and cross entropy


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784]) # create symbolic variables
Y = tf.placeholder("float", [None, 10])

w = init_weights([784, 10]) # like in linear regression, we need a shared variable weight matrix for logistic regression

py_x = model(X, w)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x, labels=Y)) # compute mean cross entropy (softmax is applied internally)
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct optimizer
# train_op2 = tf.train.AdamOptimizer(0.1).minimize(cost) # construct optimizer
predict_op = tf.argmax(py_x, 1) # at predict time, evaluate the argmax of the logistic regression

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(101):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
            # sess.run(train_op2, feed_dict={X: trX[start:end], Y: trY[start:end]})
        if i % 10 == 0:
            # Pick a random index in the list of test examples
            # and use it to compare one answer and one prediction
            rand_mnist = np.random.randint(len(teX))
            print('Random sample test |', 'Correct answer:', np.argmax(teY[rand_mnist]), '| Prediction:', sess.run(predict_op, feed_dict={X: teX})[rand_mnist])
            print('-'*20)
            print('Iteration:', i,'| Accuracy:', np.mean(np.argmax(teY, axis=1)
            == sess.run(predict_op, feed_dict={X: teX})), end='\n\n')
