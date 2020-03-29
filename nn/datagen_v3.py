import sys
from phi.flow import *


world.batch_size = 5
frames_per_sim = 16

HOW_TO = """
Generate random density and velocity fields and simulate them for %d frames (CLOSED boundary).
For each frame, "Density", "Divergence", "Pressure", "Divergent Velocity" and "Corrected Velocity" are saved.

""" % (frames_per_sim)


def random_density(shape):
    return math.maximum(0, math.randfreq(shape, power=32))


def random_velocity(shape):
    return math.randfreq(shape, power=32) * 2

class SmokeDataGen(App):

    def __init__(self):
        App.__init__(self, 'Smoke Data Generation', HOW_TO, base_dir='./data', summary='smoke_v3_highaccuracy')
        self.value_frames_per_simulation = frames_per_sim

        self.solver = SparseCG(autodiff=True, max_iterations=500, accuracy=1e-6)
        self.domain = Domain([64, 64], boundaries=CLOSED)
        self.smoke = world.add(Fluid(self.domain, density=random_density, velocity=random_velocity, batch_size=world.batch_size, buoyancy_factor=0.1), physics=IncompressibleFlow(pressure_solver=self.solver))

        # --- GUI ---
        self.div = self.smoke.domain.centered_grid(0)
        self.p = self.smoke.domain.centered_grid(0)


        self.v_div = self.smoke.domain.staggered_grid(0)
        self.v_true = self.smoke.domain.staggered_grid(0)

        self.add_field('Density', lambda: self.smoke.density)
        self.add_field('Velocity', lambda: self.smoke.velocity)
        self.add_field('Pressure', lambda: self.p)

        self.add_field('Divergent Velocity', lambda: self.v_div)
        self.add_field('Divergence', lambda: self.div.data)

        # --- Sanity Checks ---
        #self.test = self.smoke.domain.centered_grid(0)
        #self.v_test = self.smoke.domain.staggered_grid(0)

    def step(self):
        if self.steps >= self.value_frames_per_simulation:
            self.new_scene()
            self.steps = 0
            self.smoke.density = random_density
            self.smoke.velocity = random_velocity
            self.info('Starting data generation in scene %s' % self.scene)
        world.step() # simulate one step


        #get data from fluid
        self.p = self.smoke.solve_info["pressure"]
        self.div = self.smoke.solve_info["divergence"]
        self.v_div = self.smoke.solve_info['divergent_velocity']
        self.v_true = self.smoke.velocity



        # --- Save Data to Disk---
        self.scene.write_sim_frame([self.smoke.density.data, self.div.data, self.p.data, self.v_div.staggered_tensor(), self.v_true.staggered_tensor()], ["Density", "Divergence", "Pressure", "Divergent Velocity", "Corrected Velocity"], frame=self.steps)



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