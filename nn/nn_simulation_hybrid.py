# coding=utf-8
import matplotlib
from nn_architecture import *

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle
import time

DOMAIN = Domain([64, 64], boundaries=CLOSED)  # [y, x]
DESCRIPTION = u"""
Simulate random smoke distributions using a Hybrid approach where first, a trained NN predicts \n
an initial pressure guess and then the CG solver continues until the target accuracy is reached.
Performance in MS is compared to simply using the CG solver on its own.
"""

np.random.seed(2020)  # fix seed

def random_density(shape):
    return math.maximum(0, math.randfreq(shape, power=32))

def random_velocity(shape):
    return math.randfreq(shape, power=32) * 2

class NNPoissonSolver(PoissonSolver):

    def __init__(self, cg_solver=it_solver(500), normalizeInput=True):
        PoissonSolver.__init__(self, 'NN', supported_devices=('CPU', 'GPU', 'TPU'), supports_guess=False, supports_loop_counter=False, supports_continuous_masks=False)
        self.normalizeInput = normalizeInput
        self.solver = cg_solver

    # Use NN and CG Solver together to get pressure
    def solve(self, field, domain, guess):

        div = CenteredGrid(field)

        with tf.variable_scope('model'):

            # Get initial guess from NN
            guess = predict_pressure(div, normalize=self.normalizeInput)

            # Solve
            pressure, it = solve_pressure(div, domain, pressure_solver=self.solver, guess=guess)

            return pressure.data, it


class NetworkSimulation(App):

    def __init__(self):
        App.__init__(self, 'NetworkSimulation', DESCRIPTION, base_dir="NN_UNet3_Basic", record_data=False)

        # --- Set up Numerical Fluid Simulation ---
        self.solver = SparseCG(autodiff=True, max_iterations=500, accuracy=1e-3)
        self.smoke = world.add(Fluid(DOMAIN, name='smoke', density=random_density, velocity=random_velocity, buoyancy_factor=0.1), physics=IncompressibleFlow(pressure_solver=self.solver))
        self.smoke_nn = world.add(Fluid(DOMAIN, name='NN_smoke', density=self.smoke.density, velocity=self.smoke.velocity, buoyancy_factor=0.1), physics=IncompressibleFlow(pressure_solver=NNPoissonSolver(cg_solver=self.solver)))


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
        if self.frames > 0: self.total_time_cg += clock_cg

        # Hybrid Sim
        clock_hybrid = time.process_time()
        world.step(state=self.smoke_nn)  # simulate one step hybrid
        clock_hybrid = time.process_time() - clock_hybrid
        if self.frames > 0: self.total_time_hybrid += clock_hybrid

        self.frames += 1

        print("Frame %s:" % self.frames)
        print("Iterations - Hybrid: {:0} | CG: {:0}".format(self.smoke_nn.solve_info["iterations"], self.smoke.solve_info["iterations"]))
        print("Time - Hybrid: {:0.4f}ms | CG: {:0.4f}ms".format(clock_hybrid, clock_cg))
        print("Time (Avg) - Hybrid: {:0.4f}ms | CG: {:0.4f}ms".format(self.total_time_hybrid / (self.frames -1), self.total_time_cg / (self.frames -1)))



app = NetworkSimulation().prepare()
#app.action_load_model()
show(app, display=('Density Hybrid', 'Density'))
