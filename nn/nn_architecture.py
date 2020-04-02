# coding=utf-8
from phi.tf.flow import *
from phi.math import upsample2x

def pressure_unet(divergence, scope="pressure_unet"):
    """ Network structure (Based on U-Net) """
    with tf.variable_scope(scope):
        x = divergence

        print(x.shape)

        # DownConv Level 1
        c1 = tf.layers.conv2d(x, 4, 5, strides=1, activation=tf.nn.relu, padding="same", name="conv1", trainable=True,
                              reuse=tf.AUTO_REUSE)
        c2 = tf.layers.conv2d(c1, 4, 5, strides=1, activation=tf.nn.relu, padding="same", name="conv2", trainable=True,
                              reuse=tf.AUTO_REUSE)

        c3 = tf.layers.conv2d(c2, 4, 5, strides=2, activation=tf.nn.relu, padding="same", name="conv3", trainable=True,
                              reuse=tf.AUTO_REUSE)

        print(c3.shape)

        # DownConv Level 2
        c4 = tf.layers.conv2d(c3, 8, 5, strides=1, activation=tf.nn.relu, padding="same", name="conv4", trainable=True,
                              reuse=tf.AUTO_REUSE)
        c5 = tf.layers.conv2d(c4, 8, 5, strides=1, activation=tf.nn.relu, padding="same", name="conv5", trainable=True,
                              reuse=tf.AUTO_REUSE)
        c6 = tf.layers.conv2d(c5, 8, 5, strides=2, activation=tf.nn.relu, padding="same", name="conv6", trainable=True,
                              reuse=tf.AUTO_REUSE)
        print(c6.shape)

        # DownConv Level 3
        c7 = tf.layers.conv2d(c6, 16, 5, strides=1, activation=tf.nn.relu, padding="same", name="conv7", trainable=True,
                              reuse=tf.AUTO_REUSE)
        c8 = tf.layers.conv2d(c7, 16, 5, strides=1, activation=tf.nn.relu, padding="same", name="conv8", trainable=True,
                              reuse=tf.AUTO_REUSE)
        c9 = tf.layers.conv2d(c8, 16, 5, strides=2, activation=tf.nn.relu, padding="same", name="conv9", trainable=True,
                              reuse=tf.AUTO_REUSE)

        print(c9.shape)

        # Lowest Convolutions
        c10 = tf.layers.conv2d(c9, 32, 5, strides=1, activation=tf.nn.relu, padding="same", name="conv10", trainable=True,
                              reuse=tf.AUTO_REUSE)
        c11 = tf.layers.conv2d(c10, 32, 5, strides=1, activation=tf.nn.relu, padding="same", name="conv11", trainable=True,
                              reuse=tf.AUTO_REUSE)
        c12 = tf.layers.conv2d(c11, 32, 5, strides=1, activation=tf.nn.relu, padding="same", name="conv12", trainable=True,
                              reuse=tf.AUTO_REUSE)

        print(c12.shape)

        # UpConv Level 3
        u1 = upsample2x(c12)
        uc1 = tf.layers.conv2d(tf.concat([u1, c8], 3), 16, 5, strides=1, activation=tf.nn.relu, padding="same",
                               name="upconv1", trainable=True, reuse=tf.AUTO_REUSE)
        uc2 = tf.layers.conv2d(uc1, 16, 5, strides=1, activation=tf.nn.relu, padding="same", name="upconv2",
                               trainable=True, reuse=tf.AUTO_REUSE)
        uc3 = tf.layers.conv2d(uc2, 16, 5, strides=1, activation=tf.nn.relu, padding="same", name="upconv3",
                               trainable=True, reuse=tf.AUTO_REUSE)

        print(uc3.shape)

        # UpConv Level 2
        u2 = upsample2x(uc3)
        uc4 = tf.layers.conv2d(tf.concat([u2, c5], 3), 8, 5, strides=1, activation=tf.nn.relu, padding="same",
                               name="upconv4", trainable=True, reuse=tf.AUTO_REUSE)
        uc5 = tf.layers.conv2d(uc4, 8, 5, strides=1, activation=tf.nn.relu, padding="same", name="upconv5",
                               trainable=True, reuse=tf.AUTO_REUSE)
        uc6 = tf.layers.conv2d(uc5, 8, 5, strides=1, activation=tf.nn.relu, padding="same", name="upconv6",
                               trainable=True, reuse=tf.AUTO_REUSE)

        print(uc6.shape)

        # UpConv Level 1
        u3 = upsample2x(uc6)
        uc7 = tf.layers.conv2d(tf.concat([u3, c2], 3), 4, 5, strides=1, activation=tf.nn.relu, padding="same",
                               name="upconv7", trainable=True, reuse=tf.AUTO_REUSE)
        uc8 = tf.layers.conv2d(uc7, 4, 5, strides=1, activation=tf.nn.relu, padding="same", name="upconv8",
                               trainable=True, reuse=tf.AUTO_REUSE)
        uc9 = tf.layers.conv2d(uc8, 4, 5, strides=1, activation=tf.nn.relu, padding="same", name="upconv9",
                               trainable=True, reuse=tf.AUTO_REUSE)

        print(uc9.shape)

        # final Convolution
        out = tf.layers.conv2d(uc9, 1, 5, strides=1, activation=None, padding="same", name="out_conv", trainable=True,
                               reuse=tf.AUTO_REUSE)

        return out

# Predict pressure using Neural Network
def predict_pressure(divergence, normalize=True):

    if normalize:
        #mean = math.mean(divergence.data, axis=(1, 2, 3))
        #mean = math.reshape(mean, (-1, 1, 1, 1))  # reshape to broadcast correctly across batch

        #divide input by its standard deviation (normalize)
        s = math.std(divergence.data, axis=(1, 2, 3))
        s = math.reshape(s, (-1, 1, 1, 1)) # reshape to broadcast correctly across batch

        #multiply output by same factor (de-normalize)
        result = s * pressure_unet(divergence.data / s)
    else:
        result = pressure_unet(divergence.data)

    return CenteredGrid(result, divergence.box, name='pressure')

# Solver that only solves up to X iterations
def it_solver(X, acc=1e-3):
    return SparseCG(autodiff=True, max_iterations=X, accuracy=acc)
