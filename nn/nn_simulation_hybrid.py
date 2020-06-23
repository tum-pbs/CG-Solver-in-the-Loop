# coding=utf-8
import matplotlib
from nn_architecture import *

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle
import time

DOMAIN = Domain([64, 64], boundaries=CLOSED)  # [y, x]
DATAPATH = 'data/smoke_v3_test/'  # has to match DOMAIN
USE_FLOAT64 = True
DESCRIPTION = u"""
Simulate random smoke distributions using a trained NN and a normal solver for comparison.
Left: Simulation using NN as solver         Right: Simulation using numeric solver
"""

np.random.seed(2020)  # fix seed
perm = np.random.permutation(3000)


def random_density(shape):
    return math.maximum(0, math.randfreq(shape, power=32))

def random_velocity(shape):
    return math.randfreq(shape, power=32) * 2

class NNPoissonSolver(PoissonSolver):

    def __init__(self, normalizeInput=True):
        PoissonSolver.__init__(self, 'NN', supported_devices=('CPU', 'GPU', 'TPU'), supports_guess=False, supports_loop_counter=False, supports_continuous_masks=False)
        self.normalizeInput = normalizeInput

    # Use NN and CG Solver together to get pressure
    def solve(self, field, domain, guess):

        div = CenteredGrid(field)

        with tf.variable_scope('model'):

            # Get initial guess from NN
            guess = predict_pressure(div, normalize=self.normalizeInput)

            # Solve
            pressure, it = solve_pressure(div, domain, pressure_solver=it_solver(500), guess=guess)

            return pressure.data, it


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
        self.add_field('Density Hybrid', lambda: self.smoke_nn.density)
        self.add_field('Velocity', lambda: self.smoke.velocity)
        self.add_field('Velocity Hybrid', lambda: self.smoke_nn.velocity)

        self.add_field('Residuum', lambda: self.smoke.velocity.divergence())
        self.add_field('Residuum Hybrid', lambda: self.smoke_nn.velocity.divergence())
        self.add_field('Residuum Hybrid (no border)', lambda: self.smoke_nn.velocity.divergence().data[:, 1:-1, 1:-1, :])

        self.add_field('Pressure', lambda: self.smoke.solve_info.get('pressure', None))
        self.add_field('Pressure Hybrid', lambda: self.smoke_nn.solve_info.get('pressure', None))

        self.save_path = EditableString("Model Load Path", self.scene.subpath('checkpoint_%08d' % self.steps))
        self.plot_path = EditableString("Plot Data Path", self.scene.path)
        # self.save_path = EditableString("Load Path", 'NN_Unet3_Basic/nn_closed_borders/checkpoint_00300000')

        # --- PERFORMANCE ---
        self.total_time_cg = 0
        self.total_time_hybrid = 0
        self.frames = 0

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

        # CG Only Sim
        clock_cg = time.process_time()
        world.step(state=self.smoke)  # simulate one step CG-only
        clock_cg = time.process_time() - clock_cg
        self.total_time_cg += clock_cg

        # Hybrid Sim
        clock_hybrid = time.process_time()
        world.step(state=self.smoke_nn)  # simulate one step hybrid
        clock_hybrid = time.process_time() - clock_hybrid
        self.total_time_hybrid += clock_hybrid

        self.frames += 1

        self.info("Iterations - Hybrid: %s | CG: %s" % (self.smoke_nn.solve_info["iterations"], self.smoke.solve_info["iterations"]))
        self.info("Time - Hybrid: %s ms| CG: %s ms" % (clock_hybrid, clock_cg))
        self.info("Time (Avg) - Hybrid: %s ms| CG: %s ms" % (self.total_time_hybrid / self.frames, self.total_time_cg / self.frames))



app = NetworkSimulation().prepare()
#app.action_load_model()
show(app, display=('Density Hybrid', 'Density'))
