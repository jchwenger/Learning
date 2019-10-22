#!/usr/bin/env python

import tensorflow as tf

a = tf.placeholder(tf.float32) # Create a symbolic variable 'a'
b = tf.placeholder(tf.float32) # Create a symbolic variable 'b'

y = tf.multiply(a, b) # multiply the symbolic variables
z = tf.pow(a, b)
x = tf.exp(a)

with tf.Session() as sess: # create a session to evaluate the symbolic expressions
    print("Multiplication, TF variable: %f should equal 2.0" % sess.run(y, feed_dict={a: 1, b: 2})) # eval expressions with parameters for a and b
    print("Multiplication, TF variable: %f should equal 9.0" % sess.run(y, feed_dict={a: 3, b: 3}))
    print("a^b power, TF variable: %f should equal 4.0" % sess.run(z, feed_dict={a: 2, b: 2})) # eval expressions with parameters for a and b
    print("a^b power, TF variable: %f should equal 8.0" % sess.run(z, feed_dict={a: 2, b: 3}))
    print("Exponentiation e^x, TF variable: %f should equal 7.389056" % sess.run(x, feed_dict={a: 2.0}))
