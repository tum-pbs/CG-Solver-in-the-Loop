# coding=utf-8
from phi.tf.flow import *
from phi.math import upsample2x

import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DOMAIN = Domain([128, 128], boundaries=CLOSED)  # [y, x]
DATAPATH = 'data/smoke_closed/'  # has to match DOMAIN
DESCRIPTION = u"""
Simulate random smoke distributions using a trained NN and a normal solver for comparison.
Left: Simulation using NN as solver         Right: Simulation using numeric solver
"""

# Network structure (Based on U-Net)
def pressure_unet(divergence, scope="pressure_unet"):
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

# Solver that only solves up to X iterations
def it_solver(X):
    return SparseCG(autodiff=True, max_iterations=X, accuracy=1e-3)



def random_density(shape):
    return math.maximum(0, math.randfreq(shape, power=32))

def random_velocity(shape):
    return math.randfreq(shape, power=32) * 2



def divergence_free_nn(velocity, app, domain=None, obstacles=(), return_info=False):
    """
Projects the given velocity field by solving for and subtracting the pressure.
    :param return_info: if True, returns a dict holding information about the solve as a second object
    :param velocity: StaggeredGrid
    :param app: LearningApp containing the Neural Network to be used as pressure solver.
    :param domain: Domain matching the velocity field, used for boundary conditions
    :param obstacles: list of Obstacles
    :return: divergence-free velocity as StaggeredGrid
    """
    assert isinstance(velocity, StaggeredGrid)
    # --- Set up FluidDomain ---
    if domain is None:
        domain = Domain(velocity.resolution, OPEN)
    obstacle_mask = union_mask([obstacle.geometry for obstacle in obstacles])
    if obstacle_mask is not None:
        obstacle_grid = obstacle_mask.at(velocity.center_points, collapse_dimensions=False).copied_with(extrapolation='constant')
        active_mask = 1 - obstacle_grid
    else:
        active_mask = math.ones(domain.centered_shape(name='active', extrapolation='constant'))
    accessible_mask = active_mask.copied_with(extrapolation=Material.accessible_extrapolation_mode(domain.boundaries))
    fluiddomain = FluidDomain(domain, active=active_mask, accessible=accessible_mask)
    # --- Boundary Conditions, Pressure Solve ---
    velocity = fluiddomain.with_hard_boundary_conditions(velocity)
    divergence_field = velocity.divergence(physical_units=False)

    # --- Pressure Solve is done by an NN ONLY---
    pressure = CenteredGrid(app.session.run(app.nn, feed_dict={app.divergence_in: divergence_field}))

    pressure *= velocity.dx[0]
    gradp = StaggeredGrid.gradient(pressure)
    velocity -= fluiddomain.with_hard_boundary_conditions(gradp)
    return velocity if not return_info else (velocity, {'pressure': pressure, 'iterations': 0, 'divergence': divergence_field})

class IncompressibleFlowNN(Physics):
    """
Physics modelling the incompressible Navier-Stokes equations.
CAUTION: Uses a Neural Network as a pressure solver instead of a numeric one!
    """

    def __init__(self, app, make_input_divfree=False, make_output_divfree=True, conserve_density=True):
        Physics.__init__(self, [StateDependency('obstacles', 'obstacle'),
                                StateDependency('gravity', 'gravity', single_state=True),
                                StateDependency('density_effects', 'density_effect', blocking=True),
                                StateDependency('velocity_effects', 'velocity_effect', blocking=True)])
        self.app = app
        self.make_input_divfree = make_input_divfree
        self.make_output_divfree = make_output_divfree
        self.conserve_density = conserve_density

    def step(self, fluid, dt=1.0, obstacles=(), gravity=Gravity(), density_effects=(), velocity_effects=()):
        # pylint: disable-msg = arguments-differ
        gravity = gravity_tensor(gravity, fluid.rank)
        velocity = fluid.velocity
        density = fluid.density
        if self.make_input_divfree:
            velocity, fluid.solve_info = divergence_free_nn(velocity, self.app, fluid.domain, obstacles, return_info=True)
        # --- Advection ---
        density = advect.semi_lagrangian(density, velocity, dt=dt)
        velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
        if self.conserve_density and np.all(Material.solid(fluid.domain.boundaries)):
            density = density.normalized(fluid.density)
        # --- Effects ---
        for effect in density_effects:
            density = effect_applied(effect, density, dt)
        for effect in velocity_effects:
            velocity = effect_applied(effect, velocity, dt)
        velocity += buoyancy(fluid.density, gravity, fluid.buoyancy_factor) * dt
        # --- Pressure solve ---
        if self.make_output_divfree:
            velocity, fluid.solve_info = divergence_free_nn(velocity, self.app, fluid.domain, obstacles, return_info=True)
        return fluid.copied_with(density=density, velocity=velocity, age=fluid.age + dt)


class NetworkSimulation(LearningApp):

    def __init__(self):
        LearningApp.__init__(self, 'NetworkSimulation', DESCRIPTION, learning_rate=2e-4, validation_batch_size=16, training_batch_size=32, base_dir="NN_UNet3_Basic", record_data=False)

        self.value_frames_per_simulation = 100

        # --- Set up Numerical Fluid Simulation ---
        self.solver = SparseCG(autodiff=True, max_iterations=500, accuracy=1e-3)
        self.smoke = world.add(Fluid(DOMAIN, name='Smoke', density=random_density, velocity=random_velocity, buoyancy_factor=0.1), physics=IncompressibleFlow(pressure_solver=self.solver))

        # --- placeholders ---
        self.divergence_in = divergence_in = placeholder(DOMAIN.centered_shape(batch_size=None))#Network Input

        # --- Set up Neural Network Fluid Simulation ---
        with self.model_scope():
            self.nn = pressure_unet(divergence_in.data)

        self.smoke_nn = world.add(Fluid(DOMAIN, name='NN Smoke', density=self.smoke.density, velocity=self.smoke.velocity, buoyancy_factor=0.1), physics=IncompressibleFlowNN(app=self))

        # --- GUI ---
        self.add_field('Density', lambda: self.smoke.density)
        self.add_field('Density NN', lambda: self.smoke_nn.density)
        self.add_field('Velocity', lambda: self.smoke.velocity)
        self.add_field('Velocity NN', lambda: self.smoke_nn.velocity)

        self.save_path = EditableString("Load Path", self.scene.subpath('checkpoint_%08d' % self.steps))

    def action_load_model(self):
        self.load_model(self.save_path)

    def step(self):
        if self.steps >= self.value_frames_per_simulation:
            self.new_scene()
            self.steps = 0
            self.smoke.density = random_density
            self.smoke_nn.density = self.smoke.density
            self.smoke.velocity = random_velocity
            self.smoke_nn.velocity = self.smoke.velocity
        world.step() # simulate one step




# run normally with GUI
app = show(NetworkSimulation(), display=('Density NN', 'Density'))