#!/usr/bin/env python
            
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

trX = tf.reshape(x_train, [-1, 28, 28, 1])
trY = tf.one_hot(y_train, 10)
teX = tf.reshape(x_test, [-1, 28, 28, 1])
teY = tf.one_hot(y_test, 10)

batch_size = 128
test_size = 256

def init_weight(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, dimensions, filters, dropout, strides, padding='SAME'):
    input_dim = 1
    current_layer = X
    for i, dim_i in enumerate(dimensions):
        with tf.variable_scope("layer/{}".format(i)):
            W = init_weight([filters[i], 
                             filters[i], 
                             input_dim, 
                             dim_i])
            h = tf.nn.relu(
                tf.nn.conv2d(current_layer,
                             W, 
                             strides[0], 
                             padding))
            h = tf.nn.max_pool(h,
                              strides[1],
                              strides[1],
                              padding)
            h = tf.nn.dropout(h, dropout[0])
            # print('Layer', i, 'shape', h.shape)
            current_layer = h
            input_dim = dim_i

    with tf.variable_scope("flattened"):
        h = tf.contrib.layers.flatten(h)
        h = tf.nn.dropout(h, dropout[1])

        W = init_weight([h.get_shape().as_list()[1], 784])
        h = tf.nn.relu(tf.matmul(h, W))
        h = tf.nn.dropout(h, dropout[1])
        # print('Layer flattened, shape', h.shape)

    with tf.variable_scope("output"):
        W_out = init_weight([784, 10])
        h = tf.matmul(h, W_out)
        # print('Output shape', h.shape)

    return h

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

dimensions = [16, 32, 64]
filters = [3, 3, 3]

dropout = tf.placeholder("float", shape=(2))
strides = [[1,1,1,1], [1,2,2,1]]

py_x = model(X, dimensions, filters, dropout, strides)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=py_x, labels=Y))
train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # for name in [n.name for n in tf.get_default_graph().as_graph_def().node]:
    #     print(name)

    trX = trX.eval()
    trY = trY.eval()
    teX = teX.eval()
    teY = teY.eval()

    for i in range(5):
        
        training_batch = zip(range(0, len(trX), batch_size),
                            range(batch_size, len(trX)+1, batch_size))
        
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end],
                                         Y: trY[start:end],
                                         dropout: [0.8, 0.5]})

        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        
        print('Iteration:', i,
             '| Cost:', sess.run(cost, feed_dict={X: teX[test_indices],
                                                 Y: teY[test_indices],
                                                 dropout: [1.0, 1.0]}),
             '| Accuracy:', np.mean(np.argmax(teY[test_indices], axis=1) ==
                             sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                            Y: teY[test_indices],
                                                            dropout: [1.0, 1.0]})))

