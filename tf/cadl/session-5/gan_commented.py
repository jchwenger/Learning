# GAN exercise, copy & comment

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import libs.batch_norm as bn
from libs.utils import *

# Two short intros on batch norm :
# https://www.youtube.com/watch?v=nUUqwaxLnWs
# https://www.youtube.com/watch?v=dXB-KQYkzNU

# Intros on regularization
# According to the TF website: "Adding a regularization penalty over the layer
# weights and embedding weights can help prevent overfitting the training
# data." Regularization forces logits to shrink so that they are closer to the
# center of the distribution, where activation functions are closer to being
# linear, which, in turns, makes the result more linear. Applied on all layers
# of a network, it will make it overall more linear (e.g. closer to a
# polynomial of order 1 for instance, or order 2), thus reducing overfitting
# (which requires less linearity, as the networks attempts to fit all the
# irregularities of the data).
# https://www.youtube.com/watch?v=u73PU6Qwl1I
# https://www.youtube.com/watch?v=KvtGD37Rm5I
# https://www.youtube.com/watch?v=qbvRdrd0yJ8
# https://www.youtube.com/watch?v=NyG-7nRpsW8

# relu6, a variation on the widely used activation function, y = max(x, 0), 
# defined here: http://www.cs.utoronto.ca/%7Ekriz/conv-cifar10-aug2010.pdf, 
# is equal to y = min(max(x, 0), 6), which, according to the authors 
# "encourages the model to learn sparse features earlier". 
# https://www.tensorflow.org/api_docs/python/tf/nn/relu6

# phase_train: true if training, false if generating/evaluating
# reuse: False means 'create new variables', true means 'modify the existing ones'
def encoder(x, phase_train, dimensions=[], filter_sizes=[], 
        convolutional=False, activation=tf.nn.relu,
        output_activation=tf.nn.sigmoid, reuse=False):

    if convolutional:           # transforms 2D tensor
        x_tensor = to_tensor(x) # into a 4D (BWHC) one
    else:
        x_tensor = tf.reshape(
            tensor=x,
            shape=[-1, dimensions[0]])
        dimensions = dimensions[1:]

    current_input = x_tensor

    for layer_i, n_output in enumerate(dimensions):
        with tf.variable_scope(str(layer_i), reuse=reuse):
            if convolutional:
                h, W = conv2d(
                    x=current_input,
                    n_output=n_output,
                    k_h=filter_sizes[layer_i], # height, width
                    k_w=filter_sizes[layer_i], # for conv filters
                    padding='SAME', # the size of the output remains the same
                    reuse=reuse)
            else:
                h, W = linear( # = fully connected
                    x=current_input,
                    n_output=n_output,
                    reuse=reuse)

            # before activation, normalize the output 
            # (can be seen as a similar process as the activation, 
            # except it's there to make sure the data is 'smooth' 
            # throughout the network (in the same way as one 
            # normalizes the input data)
            norm = bn.batch_norm(
                x=h,
                phase_train=phase_train,
                name='bn',
                reuse=reuse)

            output = activation(norm)

        current_input = output

    flattened = flatten(current_input, name='flatten', reuse=reuse)

    if output_activation is None:
        return flattened
    else:
        return output_activation(flattened)

def decoder(z,
            phase_train,
            dimensions=[],
            channels=[],
            filter_sizes=[],
            convolutional=False,            # will be used convolutionally here
            activation=tf.nn.relu,          # why two different 
            output_activation=tf.nn.tanh,   # activation functions?
            reuse=None):

    if convolutional:
        with tf.variable_scope('fc', reuse=reuse):

            z1, W = linear( # check out the generator for an idea of
                x=z,        # what channels and dimensions look like 
                n_output=channels[0] * dimensions[0][0] * dimensions[0][1],
                reuse=reuse)

            rsz = tf.reshape(
                z1, [-1, dimensions[0][0], dimensions[0][1], channels[0]])

            current_input = activation(
                features=bn.batch_norm(
                    name='bn',
                    x=rsz,
                    phase_train=phase_train,
                    reuse=reuse))

        dimensions = dimensions[1:]
        channels = channels[1:]
        filter_sizes = filter_sizes[1:]
    else:
        current_input = z

    for layer_i, n_output in enumerate(dimensions):
        with tf.variable_scope(str(layer_i), reuse=reuse):

            if convolutional:
                h, W = deconv2d(
                    x=current_input,
                    n_output_h=n_output[0],
                    n_output_w=n_output[1],
                    n_output_ch=channels[layer_i],
                    k_h=filter_sizes[layer_i],
                    k_w=filter_sizes[layer_i],
                    padding='SAME',
                    reuse=reuse)
            else:
                h, W = linear(
                    x=current_input,
                    n_output=n_output,
                    reuse=reuse)

            # applying batch norm to all layers
            if layer_i < len(dimensions) - 1:
                norm = bn.batch_norm(
                    x=h,
                    phase_train=phase_train,
                    name='bn', reuse=reuse)
                output = activation(norm)
            else:
                output = h 
        current_input = output
        
    if output_activation is None:
        return current_input
    else:
        return output_activation(current_input) 

# G uses the decoder
def generator(z, phase_train, output_h, output_w, convolutional=True,
        n_features=32, rgb=False, reuse=None):

    n_channels = 3 if rgb else 1

    with tf.variable_scope('generator', reuse=reuse):
        return decoder(
                z=z,
                phase_train=phase_train,
                convolutional=convolutional,
                filter_sizes=[5 ,5 ,5 ,5 ,5],
                channels=[n_features * 8, n_features * 4, 
                        n_features * 2, n_features, n_channels],
                                                          
                dimensions= [                             # like in a VAE,
                        [output_h // 16, output_w // 16], # from latent 
                        [output_h // 8, output_w // 8],   # space back to image
                        [output_h // 4, output_w // 4],   # (smaller, // 16 to
                        [output_h // 2, output_w // 2],   # bigger // 2 to
                        [output_h, output_w]              # actual size)
                            ]
                if convolutional else [384, 512, n_features],
                activation=tf.nn.relu6,
                output_activation=tf.nn.tanh,
                reuse=reuse)

# The D takes an image as an input, and will use the encoder
# to project that image onto the latent space 
def discriminator(x, phase_train, convolutional=True, n_features=32, rgb=False,
        reuse=False):

    n_channels = 3 if rgb else 1

    with tf.variable_scope('discriminator', reuse=reuse):
        return encoder(
                x=x,
                phase_train=phase_train,
                convolutional=convolutional,
                filter_sizes=[5, 5, 5, 5],
                dimensions=[n_features, n_features * 2,     # like in the VAE
                            n_features * 4, n_features * 8] # gradual encoding
                if convolutional                        # into a latent space    
                else [n_features, 128, 256],
                activation=tf.nn.relu6,
                output_activation=None,
                reuse=reuse)

# where the game happens        
def GAN(input_shape, n_latent, n_features, rgb, debug=True):

    # Real input samples
    x = tf.placeholder(tf.float32, input_shape, 'x')
    x = (x / 127.5) -1.0 # x between 0 and 255, normalized to 0 - 2, then to -1 to 1
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    # I
    # ---
    # Discriminator for real input samples
    # Apply an input (x) to D, apply sigmoid to the output (logits):
    # the sigmoid outputs a number between 0 and 1, useful for a probabilistic
    # answer to our classification problem (0=fake, 1=real sample)
    # see this explanation by Ng: https://www.youtube.com/watch?v=Xvg00QnyaIY
    D_real_logits = discriminator(
        x, phase_train, n_features=n_features, rgb=rgb)
    D_real = tf.nn.sigmoid(D_real_logits)
    # D_real will be optimized directly (cf. loss below)
    # We train the discriminator to recognize real samples

    # II
    # ---
    # Generator tries to recreate input samples using latent feature vector
    # (Why no sigmoid for the output? Because we feed the output straight
    # back into the discriminator below)
    # Also, more simply: the output is an image!, not a probability
    # distribution... (We can view it in Tensorboard, cf. sum_G in Summaries)
    z = tf.placeholder(tf.float32, [None, n_latent], 'z')
    G = generator(
            z, 
            phase_train,
            output_h=input_shape[1], 
            output_w=input_shape[2],
            n_features=n_features, 
            rgb=rgb)

    # III
    # ---
    # Disciminator for generated samples
    # The output of G goes straight into the discriminator
    # (Sigmoid applied to the output, which will be optimized below)
    D_fake_logits = discriminator(
            G, 
            phase_train, 
            n_features=n_features, 
            rgb=rgb, 
            reuse=True)

    # See above (discriminator) about the presence of that sigmoid
    D_fake = tf.nn.sigmoid(D_fake_logits)

    # IV
    # ---
    # Loss is where the rules of the game are set
    with tf.variable_scope('loss'):

        # Loss functions

        # The discriminator learns to recognize real samples
        # (We compare D_real with a tensor of the same shape 
        # filled with ones (= 'real sample'), and we will ask it
        # to minimize the error (cross entropy) with it
        loss_D_real = binary_cross_entropy(
            D_real, 
            tf.ones_like(D_real), 
            name='loss_D_real')

        # The discriminator tries to categorize fake samples
        # correctly (same as above, except the comparison tensor
        # is filled with zeros (= 'fake sample')
        loss_D_fake = binary_cross_entropy(
            D_fake, 
            tf.zeros_like(D_fake), 
            name='loss_D_fake')

        # Calculate the mean loss (how often the discriminator
        # is right (combining real and fake losses)
        loss_D = tf.reduce_mean((loss_D_real + loss_D_fake) / 2)

        # The role of the generator: it optimizes itself so that
        # D categorizes its sample (output: D_fake) as real
        # samples (as close as possible to 1 = 'real samples')
        loss_G = tf.reduce_mean(binary_cross_entropy(
            D_fake, 
            tf.ones_like(D_fake), 
            name='loss_G'))

        # Summaries (for TensorBoard)
        sum_x = tf.summary.image("x", x)
        sum_D_real = tf.summary.histogram("D_real", D_real)
        sum_z = tf.summary.histogram("z", z)
        sum_G = tf.summary.image("G", G)
        sum_D_fake = tf.summary.histogram("D_fake", D_fake)

        sum_loss_D_real = tf.summary.histogram("loss_D_real", loss_D_real)
        sum_loss_D_fake = tf.summary.histogram("loss_D_fake", loss_D_fake)
        sum_loss_D = tf.summary.scalar("loss_D", loss_D)
        sum_loss_G = tf.summary.scalar("loss_G", loss_G)
        sum_D_real = tf.summary.histogram("D_real", D_real)
        sum_D_fake = tf.summary.histogram("D_fake", D_fake)

    # Return the bunch of results as a dictionary
    return {
        'loss_D': loss_D,
        'loss_G': loss_G,
        'x': x,
        'G': G,
        'z': z,
        'train': phase_train,
        'sums': {
            'G': sum_G,
            'D_real': sum_D_real,
            'D_fake': sum_D_fake,
            'loss_G': sum_loss_G,
            'loss_D': sum_loss_D,
            'loss_D_real': sum_loss_D_real,
            'loss_D_fake': sum_loss_D_fake,
            'z': sum_z,
            'x': sum_x
        }
    }



def train_ds():

    init_lr_g = 1e-4 # learning rates
    init_lr_d = 1e-4

    n_latent = 100 # still need to dig into this idea of a latent variable

    n_epochs = 1000000
    batch_size = 200
    n_samples = 15

    # Image sizes, crop etc
    input_shape = [218, 178, 3]
    crop_shape = [64, 64, 3]
    crop_factor = 0.8

    from libs.dataset_utils import create_input_pipeline
    from libs.datasets import CELEB

    files = CELEB()

    # Feed the network batch by batch, tailor images
    batch = create_input_pipeline(
            files=files,
            batch_size=batch_size,
            n_epochs=n_epochs,
            crop_shape=crop_shape,
            crop_factor=crop_factor,
            shape=input_shape)

    # [None] + crop_shape: batch (number of samples) + shape of tailored images
    gan = GAN(input_shape=[None] + crop_shape, 
                n_features=10, 
                n_latent=n_latent, 
                rgb=True, 
                debug=False)

    # List all the variables
    # Discriminator
    vars_d = [v for v in tf.trainable_variables()
              if v.name.startswith('discriminator')]
    print('Training discriminator variables:')
    [print(v.name) for v in tf.trainable_variables()
        if v.name.startswith('discriminator')]

    # Generator
    vars_g = [v for v in tf.trainable_variables()
                if v.name.startswith('generator')]
    print('Training generator variables:')
    [print(v.name) for v in tf.trainable_variables()
        if v.name.startswith('generator')]

    #########

    zs = np.random.uniform(
        -1.0, 1.0, [4, n_latent]).astype(np.float32)
    zs = make_latent_manifold(zs, n_samples)

    # Even the learning rates will be learnt! Those will be passed
    # to the opt_g & d below, which use the Adam algorithm
    lr_g = tf.placeholder(tf.float32, shape=[], name='learning_rate_g')
    lr_d = tf.placeholder(tf.float32, shape=[], name='learning_rate_d')

    # Check regularization intros above (before the code).
    # Process applied to discriminator and generator variables
    try:
        from tf.contrib.layers import apply_regularization
        d_reg = apply_regularization(
            tf.contrib.layers.l2_regularizer(1e-6), vars_d)
        g_reg = apply_regularization(
            tf.contrib.layers.l2_regularizer(1e-6), vars_g)
    except:
        d_reg, g_reg = 0, 0

    # Those two are passed to the Generator & Discriminator
    # respectively through sess.run below
    # (Both networks are trained alternatively)
    opt_g = tf.train.AdamOptimizer(lr_g, name='Adam_g').minimize(
        gan['loss_G'] + g_reg, var_list=vars_g)
    opt_d = tf.train.AdamOptimizer(lr_d, name='Adam_d').minimize(
        gan['loss_D'] + d_reg, var_list=vars_d)
    
    #########

    sess = tf.Session()
    init_op = tf.global_variables_initializer()

    #########

    # More Tensorboard summaries
    saver = tf.train.Saver()
    sums = gan['sums']

    G_sum_op = tf.summary.merge([
        sums['G'], sums['loss_G'], sums['z'],
        sums['loss_D_fake'], sums['D_fake']])
    D_sum_op = tf.summary.merge([
        sums['loss_D'], sums['loss_D_real'], sums['loss_D_fake'],
        sums['z'], sums['x'], sums['D_real'], sums['D_fake']])

    writer = tf.summary.FileWriter("./logs", sess.graph_def)

    #########

    # Multithreading / parallel calculations (if with GPU)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(init_op)

    # g = tf.get_default_graph()
    # [print(op.name) for op in g.get_operations()]

    #########

    # Checkpoint savery
    if os.path.exists("gan.ckpt"):
        saver.restore(sess, "gan.ckpt")
        print("GAN model restored.")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    step_i, t_i = 0, 0
    loss_d = 1
    loss_g = 1
    n_loss_d, total_loss_d = 1, 1
    n_loss_g, total_loss_g = 1, 1
        
    try:
        while not coord.should_stop():

            batch_xs = sess.run(batch)
            step_i += 1
            batch_zs = np.random.uniform(
                -1.0, 1.0, [batch_size, n_latent]).astype(np.float32)

            this_lr_g = min(1e-2, max(1e-6, init_lr_g * (loss_g / loss_d)**2))
            this_lr_d = min(1e-2, max(1e-6, init_lr_d * (loss_d / loss_g)**2))

            # this_lr_d *= ((1.0 - (step_i / 100000)) ** 2)
            # this_lr_g *= ((1.0 - (step_i / 100000)) ** 2)

            # 2 out of 3 steps (or according to another random-based criterium,
            # cf. the commented if & equation), we train the Discriminator,
            # and the last one we train the Generator instead
            
            # if np.random.random() > (loss_g / (loss_d + loss_g)):
            if step_i % 3 == 1:
                loss_d, _, sum_d = sess.run([gan['loss_D'], opt_d, D_sum_op], 
                                            feed_dict={gan['x']: batch_xs,
                                                        gan['z']: batch_zs,
                                                        gan['train']: True,
                                                        lr_d: this_lr_d})
                total_loss_d += loss_d
                n_loss_d += 1

                writer.add_summary(sum_d, step_i) # Tensorboard

                print('%04d d* = lr: %0.08f, loss: %08.06f, \t' %
                    (step_i, this_lr_d, loss_d) +
                    'g  = lr: %0.08f, loss: %08.06f' % (this_lr_g, loss_g))

            else:

                loss_g, _, sum_g = sess.run([gan['loss_G'], opt_g, G_sum_op],
                                            feed_dict={gan['z']: batch_zs,
                                                        gan['train']: True,
                                                        lr_g: this_lr_g})
                total_loss_g += loss_g
                n_loss_g += 1

                writer.add_summary(sum_g, step_i) # Tensorboard

                print('%04d d  = lr: %0.08f, loss: %08.06f, \t' %
                        (step_i, this_lr_d, loss_d) +
                        'g* = lr: %0.08f, loss: %08.06f' % (this_lr_g, loss_g))

            if step_i % 100 == 0:

                samples = sess.run(gan['G'], feed_dict={
                                                gan['z']: zs,
                                                gan['train']: False})

                # Create a wall of images of the latent space (what the
                # network learns)
                montage(np.clip((samples + 1) * 127.5, 0, 255).astype(np.uint8), 
                        'imgs/gan_%08d.png' % t_i)
                t_i += 1

                print('generator loss:', total_loss_g / n_loss_g)
                print('discriminator loss:', total_loss_d / n_loss_d)

                # Save variable to disk
                save_path = saver.save(sess, "./gan.ckpt",
                                        global_step=step_i,
                                        write_meta_graph=False)
                print("Model saved in file: %s" % save_path)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # One thread issued an exception, let them all shut down
        coord.request_stop()

    # Wait for them to finish
    coord.join(threads)

    # Clean up
    sess.close()


if __name__ == '__main__':
    train_ds()
