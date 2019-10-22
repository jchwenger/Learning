#!/usr/bin/env python

import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

trX = tf.reshape(x_train, [-1, 784])
trY = tf.one_hot(y_train, 10)
teX = tf.reshape(x_test, [-1, 784])
teY = tf.one_hot(y_test, 10)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_dropout():
    return tf.Variable("float")

def model(X, weights, dropout): 

    current_input = X
    current_input = tf.nn.dropout(X, dropout[0])

    for i in range(len(weights[:-1])):
        h = tf.nn.relu(tf.matmul(current_input, weights[i]))
        h = tf.nn.dropout(h, dropout[1])
        current_input = h

    return tf.matmul(current_input, weights[-1])

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

layers = [784, 128, 64, 64, 64, 128, 10]

weights = []

for i in range(len(layers)-1):
    weight_i = init_weights([layers[i], layers[i+1]])
    weights.append(weight_i)

dropout = tf.placeholder("float", shape=(2))

py_x = model(X, weights, dropout)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x, labels=Y))
# train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    trX = trX.eval()
    trY = trY.eval()
    teX = teX.eval()
    teY = teY.eval()

    for i in range(20):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):

            sess.run(train_op, feed_dict={X: trX[start:end], 
                                          Y: trY[start:end],
                                          dropout: [0.8, 0.5]})

        if i % 2 == 0:
            cost_i = sess.run(cost, feed_dict={X: teX,
                                               Y: teY,
                                               dropout: [0.8, 0.5]}) 

            acc_i = np.mean(np.argmax(teY, axis=1) ==
                           sess.run(predict_op, feed_dict={X: teX, 
                                                           Y: teY,
                                                           dropout: [1.0, 1.0]}))

            print('Iterations:', i, 
                  '| Cost:', cost_i,
                  '| Accuracy:', acc_i)
