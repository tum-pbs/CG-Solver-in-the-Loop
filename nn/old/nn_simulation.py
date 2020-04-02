# coding=utf-8
import matplotlib

from phi.tf.flow import *
from phi.math import upsample2x

matplotlib.use('Agg')

DOMAIN = Domain([64, 64], boundaries=CLOSED)  # [y, x]
DATAPATH = 'data/smoke_closed/'  # has to match DOMAIN
DESCRIPTION = u"""
Simulate random smoke distributions using a trained NN and a normal solver for comparison.
Left: Simulation using NN as solver         Right: Simulation using numeric solver
"""


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


def random_density(shape):
    return math.maximum(0, math.randfreq(shape, power=32))


def random_velocity(shape):
    return math.randfreq(shape, power=32) * 2


class NNPoissonSolver(PoissonSolver):

    def __init__(self):
        PoissonSolver.__init__(self, 'NN', supported_devices=('CPU', 'GPU', 'TPU'), supports_guess=False, supports_loop_counter=False, supports_continuous_masks=False)

    def solve(self, field, domain, guess):
        with tf.variable_scope('model'):
            return pressure_unet(field), None


class NetworkSimulation(App):

    def __init__(self):
        App.__init__(self, 'NetworkSimulation', DESCRIPTION, base_dir="NN_UNet3_Basic", record_data=False)

        # --- Set up Numerical Fluid Simulation ---
        self.solver = SparseCG(autodiff=True, max_iterations=500, accuracy=1e-3)
        self.smoke = world.add(Fluid(DOMAIN, name='smoke', density=random_density, velocity=random_velocity, buoyancy_factor=0.1), physics=IncompressibleFlow(pressure_solver=self.solver))
        self.smoke_nn = world.add(Fluid(DOMAIN, name='NN_smoke', density=self.smoke.density, velocity=self.smoke.velocity, buoyancy_factor=0.1), physics=IncompressibleFlow(pressure_solver=NNPoissonSolver()))

       # --- GUI ---
        self.value_frames_per_simulation = 100
        self.add_field('Density', lambda: self.smoke.density)
        self.add_field('Density NN', lambda: self.smoke_nn.density)
        self.add_field('Velocity', lambda: self.smoke.velocity)
        self.add_field('Velocity NN', lambda: self.smoke_nn.velocity)

        self.add_field('Residuum', lambda: self.smoke.velocity.divergence())
        self.add_field('Residuum NN', lambda: self.smoke_nn.velocity.divergence())

        self.save_path = EditableString("Load Path", self.scene.subpath('checkpoint_%08d' % self.steps))
        # self.save_path = EditableString("Load Path", 'NN_Unet3_Basic/nn_closed_borders/checkpoint_00300000')

    def action_load_model(self):
        self.session.restore(self.save_path, scope='model')

    def step(self):
        if self.steps >= self.value_frames_per_simulation:
            self.new_scene()
            self.steps = 0
            self.smoke.density = random_density
            self.smoke_nn.density = self.smoke.density
            self.smoke.velocity = random_velocity
            self.smoke_nn.velocity = self.smoke.velocity
        world.step()  # simulate one step


app = NetworkSimulation().prepare()
#app.action_load_model()
show(app, display=('Density NN', 'Density'))
