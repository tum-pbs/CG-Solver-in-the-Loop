from phi.flow import *


world.batch_size = 4


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
        App.__init__(self, 'Smoke Data Generation', HOW_TO, base_dir='~/phi/data', summary='smoke')
        self.value_frames_per_simulation = 16

        self.solver = SparseCG(autodiff=True, max_iterations=500, accuracy=1e-3)
        self.smoke = world.add(Fluid(Domain([64, 64]), density=random_density, velocity=random_velocity, batch_size=world.batch_size, buoyancy_factor=0.1), physics=IncompressibleFlow(pressure_solver=self.solver))


        #GUI
        self.div = self.smoke.domain.centered_grid(0)
        self.div_pre = self.smoke.domain.centered_grid(0)
        self.p = self.smoke.domain.centered_grid(0)
        self.p_pre = self.smoke.domain.centered_grid(0)
        self.p_div_pre = self.smoke.domain.centered_grid(0)

        self.add_field('Density', lambda: self.smoke.density)
        self.add_field('Velocity', lambda: self.smoke.velocity)
        self.add_field('Divergence', lambda: self.div)
        self.add_field('Divergence (Preprocessed)', lambda: self.div_pre)
        self.add_field('Pressure', lambda: self.p)
        self.add_field('Pressure (Preprocessed)', lambda: self.p_pre)
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
        mean = np.mean(self.div)
        self.div_pre = self.div - mean
        normalization_factor = np.percentile(math.abs(self.div_pre), 95)
        self.div_pre = self.div_pre / normalization_factor

        self.p_pre = (self.p - mean) / normalization_factor

        press, _ = solve_pressure(self.smoke.domain.centered_grid(self.div_pre), self.smoke.domain, pressure_solver=self.solver)
        self.p_div_pre = press.data


        # save preprocessed data to disk
        #self.scene.write_sim_frame([self.density_data.data, self.velocity_data.staggered_tensor(), self.div_data.data, self.p_data.data], ["Density", "Velocity", "Div_Data", "P_Data"], frame=self.steps)


show(SmokeDataGen(), display=('Density', 'Velocity'))
