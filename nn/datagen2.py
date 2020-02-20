import sys
from phi.flow import *


world.batch_size = 5


HOW_TO = """
Press 'Play' to continuously generate data until stopped.
Each step computes one frame for each scene in the batch (batch_size=%d).

The number of frames per simulation can be adjusted in the model parameters section.

The text box next to 'Play' lets you choose how many frames you want to generate in total. It should be a multiple of the frames per simulation.
""" % world.batch_size


def random_density(shape):
    return math.maximum(0, math.randfreq(shape, power=32))


def random_velocity(shape):
    return math.randfreq(shape, power=32) * 2


class SmokeDataGen(App):

    def __init__(self):
        App.__init__(self, 'Smoke Data Generation', HOW_TO, base_dir='./data', summary='smoke2')
        self.value_frames_per_simulation = 16

        self.solver = SparseCG(autodiff=True, max_iterations=500, accuracy=1e-3)
        self.smoke = world.add(Fluid(Domain([64, 64]), density=random_density, velocity=random_velocity, batch_size=world.batch_size, buoyancy_factor=0.1), physics=IncompressibleFlow(pressure_solver=self.solver))


        #GUI
        self.div = self.smoke.domain.centered_grid(0)
        self.div_pre = self.smoke.domain.centered_grid(0)

        self.p = self.smoke.domain.centered_grid(0)
        self.p_div_pre = self.smoke.domain.centered_grid(0)

        self.add_field('Density', lambda: self.smoke.density)
        self.add_field('Velocity', lambda: self.smoke.velocity)
        self.add_field('Divergence', lambda: self.div)
        self.add_field('Divergence (Preprocessed)', lambda: self.div_pre)
        self.add_field('Pressure', lambda: self.p)
        self.add_field('Pressure (of Preprocessed Div)', lambda: self.p_div_pre)

    def step(self):
        if self.steps >= self.value_frames_per_simulation:
            self.new_scene()
            self.steps = 0
            self.smoke.density = random_density
            self.smoke.velocity = random_velocity
            self.info('Starting data generation in scene %s' % self.scene)
        world.step() # simulate one step

        #get data from fluid
        density_data = self.smoke.density.data
        velocity_data = self.smoke.velocity.data
        self.p = self.smoke.solve_info["pressure"].data
        self.div = self.smoke.solve_info["divergence"].data

        #preprocess data
        self.div_pre = self.div - np.mean(self.div)
        self.div_pre = self.div_pre / np.percentile(math.abs(self.div_pre), 95)

        press, _ = solve_pressure(self.smoke.domain.centered_grid(self.div_pre), self.smoke.domain, pressure_solver=self.solver)
        self.p_div_pre = press.data


        # save preprocessed data to disk
        self.scene.write_sim_frame([self.smoke.density.data, self.div_pre, self.p_div_pre], ["Density", "Divergence", "Pressure"], frame=self.steps)



# run from command line with fixed amount of steps
if len(sys.argv) > 1:
    sims_to_generate = int(sys.argv[1])

    app = SmokeDataGen()
    steps_to_run = (sims_to_generate / world.batch_size) * app.value_frames_per_simulation

    app.info('Started Data Generation!')
    app.info('Generating %s simulations...' % sims_to_generate)

    app.play(max_steps=steps_to_run)

# run normally with GUI
else:
    app = show(SmokeDataGen(), display=('Density', 'Velocity'))