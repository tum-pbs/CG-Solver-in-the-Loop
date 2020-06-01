# coding=utf-8
import matplotlib
from nn_architecture import *

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle

DOMAIN = Domain([64, 64], boundaries=CLOSED)  # [y, x]
DATAPATH = 'data/smoke_v3_test/'  # has to match DOMAIN
USE_FLOAT64 = True
DESCRIPTION = u"""
Simulate random smoke distributions using a trained NN and a normal solver for comparison.
Left: Simulation using NN as solver         Right: Simulation using numeric solver
"""

np.random.seed(2020)  # fix seed
perm = np.random.permutation(3000)

# create a geometry mask that is 0: at the border and 1: everywhere else
g_mask = math.ones(DOMAIN.centered_shape())
g_mask.data[:, 0, :, :] = 0
g_mask.data[:, :, 0, :] = 0
g_mask.data[:, :, -1, :] = 0
g_mask.data[:, -1, :, :] = 0


def random_density(shape):
    return math.maximum(0, math.randfreq(shape, power=32))


def random_velocity(shape):
    return math.randfreq(shape, power=32) * 2


class NNPoissonSolver(PoissonSolver):

    def __init__(self, normalizeInput=True):
        PoissonSolver.__init__(self, 'NN', supported_devices=('CPU', 'GPU', 'TPU'), supports_guess=False, supports_loop_counter=False, supports_continuous_masks=False)
        self.normalizeInput = normalizeInput

    # this solver normalizes the input, feeds it into the NN and de-normalizes the output again
    def solve(self, field, domain, guess):
        with tf.variable_scope('model'):

            result = predict_pressure(CenteredGrid(field), normalize=self.normalizeInput)
            return result.data, None


class NetworkSimulation(App):

    def __init__(self):
        App.__init__(self, 'NetworkSimulation', DESCRIPTION, base_dir="NN_UNet3_Basic", record_data=False)

        # --- Set up Numerical Fluid Simulation ---
        self.solver = SparseCG(autodiff=True, max_iterations=500, accuracy=1e-3)
        self.smoke = world.add(Fluid(DOMAIN, name='smoke', density=random_density, velocity=random_velocity, buoyancy_factor=0.1), physics=IncompressibleFlow(pressure_solver=self.solver))
        self.smoke_nn = world.add(Fluid(DOMAIN, name='NN_smoke', density=self.smoke.density, velocity=self.smoke.velocity, buoyancy_factor=0.1), physics=IncompressibleFlow(pressure_solver=NNPoissonSolver()))


        #placeholders
        self.divergence_in = placeholder(DOMAIN.centered_shape(batch_size=None))#Network Input
        self.true_pressure = placeholder(DOMAIN.centered_shape(batch_size=None))  # Ground Truth

        self.zero_guess = placeholder(DOMAIN.centered_shape(batch_size=None))  # Zero Pressure Guess

        self.max_it = self.editable_int("Max_Iterations", 500, (500,500))# Only used for manual plotting
        self.accuracy = self.editable_float("Accuracy", 1e-3)# Only used for manual plotting

        # --- Load Test Set ---
        test_set = Dataset.load(DATAPATH, range(0, 2999), name='testset')
        self.data_reader = BatchReader(test_set, ("Divergence", "Pressure"))

        with tf.variable_scope('model'):
            self.pred_pressure = predict_pressure(self.divergence_in)  # NN Pressure Guess

            div = math.to_float(self.divergence_in, float64=USE_FLOAT64)
            p_pred = math.to_float(self.pred_pressure, float64=USE_FLOAT64)
            p_true = math.to_float(self.true_pressure, float64=USE_FLOAT64)
            p_zero = math.to_float(self.zero_guess, float64=USE_FLOAT64)
            acc = math.to_float(self.accuracy, float64=USE_FLOAT64)

            # Pressure Solves with different Guesses (max iterations as placeholder)
            self.p_predGuess, self.iter_guess = solve_pressure(div, DOMAIN, pressure_solver=it_solver(self.max_it, acc), guess=p_pred)
            self.p_trueGuess, self.iter_true = solve_pressure(div, DOMAIN, pressure_solver=it_solver(self.max_it, acc), guess=p_true)
            self.p_noGuess, self.iter_zero = solve_pressure(div, DOMAIN, pressure_solver=it_solver(self.max_it, acc), guess=p_zero)

            # Netwrok prediction + limited iterations
            self.p_pred_i1, _ = solve_pressure(div, DOMAIN, pressure_solver=it_solver(1), guess=p_pred)

        # --- GUI ---
        self.value_frames_per_simulation = 100
        self.add_field('Density', lambda: self.smoke.density)
        self.add_field('Density NN', lambda: self.smoke_nn.density)
        self.add_field('Velocity', lambda: self.smoke.velocity)
        self.add_field('Velocity NN', lambda: self.smoke_nn.velocity)

        self.add_field('Residuum', lambda: self.smoke.velocity.divergence())
        self.add_field('Residuum NN', lambda: self.smoke_nn.velocity.divergence())
        self.add_field('Residuum NN (no border)', lambda: self.smoke_nn.velocity.divergence().data[:, 1:-1, 1:-1, :])

        self.add_field('SDF', lambda: SDF(DOMAIN))

        self.add_field('Pressure', lambda: self.smoke.solve_info.get('pressure', None))
        self.add_field('Pressure NN', lambda: self.smoke_nn.solve_info.get('pressure', None))

        self.save_path = EditableString("Model Load Path", self.scene.subpath('checkpoint_%08d' % self.steps))
        self.plot_path = EditableString("Plot Data Path", self.scene.path)
        # self.save_path = EditableString("Load Path", 'NN_Unet3_Basic/nn_closed_borders/checkpoint_00300000')

    def action_load_model(self):
        self.session.restore(self.save_path, scope='model')

    def action_calculate_iterations_plot(self):
        self.info('Calculate Accuracy/Iterations Plot Data...')

        batch = perm[:100]
        accuracies = (1e-1, 0.5e-1, 1e-2, 0.5e-2, 1e-3, 0.5e-3, 1e-4, 0.5e-4, 1e-5, 0.5e-5, 1e-6)

        mean_it_zero = []
        mean_it_true = []
        mean_it_pred = []

        for accuracy in accuracies:
            iterations_zero = []
            iterations_guess = []
            iterations_true = []

            self.info('Calculating for %s...' % accuracy)

            # Look at each sample individually
            for i in batch:
                sample = self.data_reader[int(i)]
                zero = np.zeros_like(sample[0])

                f_dict = {self.divergence_in.data: sample[0], self.true_pressure.data: sample[1], self.zero_guess.data: zero, self.max_it: 2000, self.accuracy: accuracy}

                # Solve for pressure for this sample using different guesses
                itZero, itPred, itTrue = self.session.run([self.iter_zero, self.iter_guess, self.iter_true], f_dict)

                iterations_zero.append(itZero)
                iterations_guess.append(itPred)
                iterations_true.append(itTrue)

            mean_it_zero.append(math.mean(iterations_zero))
            mean_it_true.append(math.mean(iterations_true))
            mean_it_pred.append(math.mean(iterations_guess))

        # Save Calculated data to disk for later plotting
        with open(self.plot_path + '/model_iter.data', 'wb') as file:
            pickle.dump(mean_it_pred, file)

        with open(self.plot_path + '/zeroguess_iter.data', 'wb') as file:
            pickle.dump(mean_it_zero, file)

        with open(self.plot_path + '/trueguess_iter.data', 'wb') as file:
            pickle.dump(mean_it_true, file)

        with open(self.plot_path + '/acc.data', 'wb') as file:
            pickle.dump(accuracies, file)

        self.info('Finished calculation, saved plot data to %s' % self.plot_path)

    def action_save_guess_images(self):

        self.info('Generate Pressure Guess Image...')

        batch_size = 20
        batch = self.data_reader[perm[:batch_size]]
        f_dict = {self.divergence_in.data: batch[0], self.true_pressure: batch[1]}

        div, p_pred = self.session.run([self.divergence_in, self.pred_pressure], feed_dict=f_dict)
        p = p_pred.data
        p_true = batch[1]

        #res = math.abs(div - p_pred.laplace()).data[:, 1:-1, 1:-1, :]
        res = math.abs(div - p_pred.laplace()).data

        model_images = []
        true_images = []

        for i in range(batch_size):
            p_image = np.reshape(p[i], DOMAIN.resolution)
            #p_image = p_image - np.mean(p_image)
            p_true_image = np.reshape(p_true[i], DOMAIN.resolution)
            #residuum = np.reshape(res[i], DOMAIN.resolution - [2, 2])
            residuum = np.reshape(res[i], DOMAIN.resolution)

            model_images.append(p_image)
            true_images.append(p_true_image)

            plt.imshow(p_image, vmin=-10.0, vmax=10.0, cmap='bwr', origin='lower')
            path = self.scene.subpath(name='pressure_nn_%s' % i)
            plt.savefig(path, dpi=200)
            plt.close()

            plt.imshow(p_true_image, vmin=-10.0, vmax=10.0, cmap='bwr', origin='lower')
            path = self.scene.subpath(name='pressure_true_%s' % i)
            plt.savefig(path, dpi=200)
            plt.close()

            plt.imshow(residuum, cmap='bwr', origin='lower')
            path = self.scene.subpath(name='residuum_nn_%s' % i)
            plt.savefig(path, dpi=200)
            plt.close()

        np.save(self.plot_path + "/model_images", arr=model_images)
        np.save(self.plot_path + "/true_images", arr=true_images)
        self.info('Saved Pressure Guess images to %s' % path)

    #Use matplotlib to make diagram of residuum mean/max vs. iterations with and without guess
    def action_calculate_residuum_plot(self):
        residuum_max = []
        residuum_mean = []
        residuum_max_noguess = []
        residuum_mean_noguess = []

        all_maxima = []
        all_means = []
        all_maxima_noguess = []
        all_means_noguess = []
        it = []
        it_to_plot = 200
        batch_size = 100

        batch = self.data_reader[perm[:batch_size]]
        zero = np.zeros_like(batch[0])

        f_dict = {self.divergence_in.data: batch[0], self.true_pressure.data: batch[1], self.zero_guess.data: zero}
        f_dict[self.accuracy] = 1e-12  # very high accuracy so it can solve "as long as possible"

        self.info('Calculate Residuum Plot data...')

        for i in range(0, it_to_plot):
            f_dict[self.max_it] = i# set max_it to current i

            if i == 0:
                #take guess itself for 0th iteration
                div_in, pressure = self.session.run([self.divergence_in, self.pred_pressure], f_dict)
                pressure_noguess = DOMAIN.centered_grid(0)

            else:
                #solve for pressure with and without predicted guess
                div_in, pressure, pressure_noguess = self.session.run([self.divergence_in, self.p_predGuess, self.p_noGuess], f_dict)

            it.extend([i] * batch_size)

            #residuum with guess (absolute value)
            #residuum = math.abs(div_in - pressure.laplace()).data[:, 1:-1, 1:-1, :]
            residuum = math.abs(div_in - pressure.laplace()).data
            batch_maxima = math.max(residuum, axis=(1, 2, 3))
            batch_means = math.mean(residuum, axis=(1, 2, 3))

            #record individual max/mean for scatterplot
            all_maxima.extend(batch_maxima)
            all_means.extend(batch_means)

            #record average mean/max over batch for curve plot
            residuum_max.append(math.mean(batch_maxima))
            residuum_mean.append(math.mean(batch_means))




            # residuum without guess (absolute value)
            #residuum_noguess = math.abs(div_in - pressure_noguess.laplace()).data[:, 1:-1, 1:-1, :]
            residuum_noguess = math.abs(div_in - pressure_noguess.laplace()).data
            batch_maxima_noguess = math.max(residuum_noguess, axis=(1, 2, 3))
            batch_means_noguess = math.mean(residuum_noguess, axis=(1, 2, 3))

            #record individual max/mean for scatterplot
            all_maxima_noguess.extend(batch_maxima_noguess)
            all_means_noguess.extend(batch_means_noguess)

            residuum_max_noguess.append(math.mean(batch_maxima_noguess))
            residuum_mean_noguess.append(math.mean(batch_means_noguess))

            print(i)

        # Save Calculated data to disk for later plotting
        with open(self.plot_path + '/supervised_full_resmax.data', 'wb') as file:
            pickle.dump([range(0, it_to_plot), residuum_max], file)

        with open(self.plot_path + '/supervised_full_resmean.data', 'wb') as file:
            pickle.dump([range(0, it_to_plot), residuum_mean], file)

        with open(self.plot_path + '/supervised_full_points.data', 'wb') as file:
            pickle.dump([it, all_means, all_maxima], file)


        with open(self.plot_path + '/zeroguess_resmax.data', 'wb') as file:
            pickle.dump([range(0, it_to_plot), residuum_max_noguess], file)

        with open(self.plot_path + '/zeroguess_resmean.data', 'wb') as file:
            pickle.dump([range(0, it_to_plot), residuum_mean_noguess], file)

        with open(self.plot_path + '/zeroguess_points.data', 'wb') as file:
            pickle.dump([it, all_means_noguess, all_maxima_noguess], file)


        self.info('Finished calculation, saved plot data to %s' % self.plot_path)

    def action_residuum_images(self):

        self.info('Generate Residuum Image with loaded Model...')

        batch_size = 20
        batch = self.data_reader[perm[:batch_size]]
        f_dict = {self.divergence_in.data: batch[0], self.true_pressure: batch[1]}

        self.info('Evaluate Network Pressure Prediction')
        div, p_pred, p_pred_i1 = self.session.run([self.divergence_in, self.pred_pressure, self.p_pred_i1], feed_dict=f_dict)

        # calculate residuum
        res = math.abs(div - p_pred.laplace()).data
        res_i1 = math.abs(div - p_pred_i1.laplace()).data

        res_images = []
        res_i1_images = []

        self.info('Plot Images')
        for i in range(batch_size):
            residuum = np.reshape(res[i], DOMAIN.resolution)
            residuum_i1 = np.reshape(res_i1[i], DOMAIN.resolution)

            res_images.append(residuum)
            res_i1_images.append(residuum_i1)

            plt.imshow(residuum, cmap='bwr', origin='lower')
            path = self.scene.subpath(name='residuum_nn_%s' % i)
            plt.savefig(path, dpi=200)
            plt.close()

            plt.imshow(residuum_i1, cmap='bwr', origin='lower')
            path = self.scene.subpath(name='residuum_nn_i1_%s' % i)
            plt.savefig(path, dpi=200)
            plt.close()


        self.info('Save Residuum Images')
        np.save(self.plot_path + "/res_images", arr=res_images)
        np.save(self.plot_path + "/res_i1_images", arr=res_i1_images)
        self.info('Saved Residuum images to %s' % path)


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
