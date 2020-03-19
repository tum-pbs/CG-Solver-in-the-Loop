import sys
from phi.flow import *


world.batch_size = 5
frames_per_sim = 16

HOW_TO = """
Generate random density and velocity fields and simulate them for %d frames (CLOSED boundary).
For each frame, "Density", "Divergence", "Pressure", "Advected Velocity" and "Corrected Velocity" are saved.

"Density": The density field as it has been advected in the simulation.
"Advected Velocity": The divergent velocity field (before being corrected).
"Divergence": The divergence of "Advected Velocity", divided by its standard deviation (for normalization).
"Pressure": The pressure used to correct "Advected Velocity" to "Corrected Velocity", divided by the same factor as "Divergence"
"Corrected Velocity": The velocity field after being corrected with "Pressure"

When solving for pressure with "Divergence" as input, the result must be divided by the standard deviation of "Advected Velocity".
(De-normalization)

""" % (frames_per_sim)


def random_density(shape):
    return math.maximum(0, math.randfreq(shape, power=32))


def random_velocity(shape):
    return math.randfreq(shape, power=32) * 2

def correct(velocity, pressure, denormalize=False):

    if denormalize:
        div = velocity.divergence(physical_units=False)
        factor = math.std(div.data, axis=(1, 2, 3))
        factor = math.reshape(factor, (-1, 1, 1, 1))  # reshape to broadcast correctly across batch

        pressure = pressure * factor

    gradp = StaggeredGrid.gradient(pressure)
    return (velocity - gradp)

class SmokeDataGen(App):

    def __init__(self):
        App.__init__(self, 'Smoke Data Generation', HOW_TO, base_dir='./data', summary='smoke_test')
        self.value_frames_per_simulation = frames_per_sim

        self.solver = SparseCG(autodiff=True, max_iterations=500, accuracy=1e-3)
        self.domain = Domain([64, 64], boundaries=CLOSED)
        self.smoke = world.add(Fluid(self.domain, density=random_density, velocity=random_velocity, batch_size=world.batch_size, buoyancy_factor=0.1), physics=IncompressibleFlow(pressure_solver=self.solver))

        # --- GUI ---
        self.div = self.smoke.domain.centered_grid(0)
        self.p = self.smoke.domain.centered_grid(0)

        self.div_pre = self.smoke.domain.centered_grid(0)
        self.p_pre = self.smoke.domain.centered_grid(0)


        self.v_div = self.smoke.domain.staggered_grid(0)
        self.v_true = self.smoke.domain.staggered_grid(0)

        self.add_field('Density', lambda: self.smoke.density)
        self.add_field('Velocity', lambda: self.smoke.velocity)
        self.add_field('Pressure', lambda: self.p)
        self.add_field('Pressure (of normalized Divergence)', lambda: self.p_pre)

        self.add_field('Divergent Velocity', lambda: self.v_div)
        self.add_field('Divergence', lambda: self.div.data)
        self.add_field('Divergence (Normalized)', lambda: self.div_pre.data)

        # --- Sanity Checks ---
        self.test = self.smoke.domain.centered_grid(0)
        self.v_test = self.smoke.domain.staggered_grid(0)
        self.add_field('Test', lambda: self.test)
        self.add_field('Test Vel', lambda: self.v_test)

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
        self.v_div = self.smoke.solve_info['advected velocity']
        self.v_true = self.smoke.velocity



        # --- Preprocess Data ---
        normalization_factor = math.std(self.div.data, axis=(1, 2, 3))
        normalization_factor = math.reshape(normalization_factor, (-1, 1, 1, 1)) # reshape to broadcast correctly across batch

        normalization_factor = 0.1

        self.div_pre = self.div / normalization_factor
        self.p_pre = self.p / normalization_factor

        print (normalization_factor)


        # --- Sanity Checks ---
        div_pre_copy = self.domain.centered_grid(np.copy(self.div_pre.data))
        p_resolve, it = solve_pressure(div_pre_copy, self.smoke.domain, pressure_solver=self.solver, guess=self.p_pre.data)

        print("Iterations needed (Normalized): %d" % it)
        np.testing.assert_equal(p_resolve.data, self.p_pre.data)




        div_copy = self.domain.centered_grid(np.copy(self.div.data))
        p_2, it = solve_pressure(div_copy, self.smoke.domain, pressure_solver=self.solver, guess=self.p.data)

        print("Iterations needed (Normal): %d" % it)
        np.testing.assert_equal(p_2.data, self.p.data)




        self.test = p_resolve
        self.v_test = correct(self.v_div, self.p_pre, denormalize=True)



        # --- Save Data to Disk---
        self.scene.write_sim_frame([self.smoke.density.data, self.div_pre.data, self.p_pre.data, self.v_div.staggered_tensor(), self.v_true.staggered_tensor()], ["Density", "Divergence", "Pressure", "Advected Velocity", "Corrected Velocity"], frame=self.steps)



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