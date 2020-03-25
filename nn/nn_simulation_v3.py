# coding=utf-8
import matplotlib

from phi.tf.flow import *
from phi.math import upsample2x

matplotlib.use('Agg')
import matplotlib.pyplot as plt

DOMAIN = Domain([64, 64], boundaries=CLOSED)  # [y, x]
DATAPATH = 'data/smoke_v3_test/'  # has to match DOMAIN
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

# Solver that only solves up to X iterations
def it_solver(X):
    return SparseCG(autodiff=True, max_iterations=X, accuracy=1e-3)

# Predict pressure using Neural Network
def predict_pressure(divergence, normalize=True):

    if normalize:
        #divide input by its standard deviation (normalize)
        s = math.std(divergence.data, axis=(1, 2, 3))
        s = math.reshape(s, (-1, 1, 1, 1)) # reshape to broadcast correctly across batch

        #multiply output by same factor (de-normalize)
        result =  s * pressure_unet(divergence.data / s)
    else:
        result = pressure_unet(divergence.data)

    return CenteredGrid(result, divergence.box, name='pressure')


class NNPoissonSolver(PoissonSolver):

    def __init__(self, normalizeInput=True):
        PoissonSolver.__init__(self, 'NN', supported_devices=('CPU', 'GPU', 'TPU'), supports_guess=False, supports_loop_counter=False, supports_continuous_masks=False)
        self.normalizeInput = normalizeInput

    # this solver normalizes the input, feeds it into the NN and de-normalizes the output again
    def solve(self, field, domain, guess):
        with tf.variable_scope('model'):

            if self.normalizeInput:
                # divide input by its standard deviation (normalize)
                s = math.std(field, axis=(1, 2, 3))
                s = math.reshape(s, (-1, 1, 1, 1))  # reshape to broadcast correctly across batch

                # multiply output by same factor (de-normalize)
                result = s * pressure_unet(field / s)
            else:
                result = pressure_unet(field)

            return result, None


class NetworkSimulation(App):

    def __init__(self):
        App.__init__(self, 'NetworkSimulation', DESCRIPTION, base_dir="NN_UNet3_Basic", record_data=False)

        # --- Set up Numerical Fluid Simulation ---
        self.solver = SparseCG(autodiff=True, max_iterations=500, accuracy=1e-3)
        self.smoke = world.add(Fluid(DOMAIN, name='smoke', density=random_density, velocity=random_velocity, buoyancy_factor=0.1), physics=IncompressibleFlow(pressure_solver=self.solver))
        self.smoke_nn = world.add(Fluid(DOMAIN, name='NN_smoke', density=self.smoke.density, velocity=self.smoke.velocity, buoyancy_factor=0.1), physics=IncompressibleFlow(pressure_solver=NNPoissonSolver(normalizeInput=False)))


        #placeholders
        self.divergence_in = placeholder(DOMAIN.centered_shape(batch_size=None))#Network Input
        self.true_pressure = placeholder(DOMAIN.centered_shape(batch_size=None))  # Ground Truth
        self.max_it = self.editable_int("Max_Iterations", 500, (500, 500))  # Only used for manual plotting

        # --- Load Test Set ---
        test_set = Dataset.load(DATAPATH, range(0, 2999), name='testset')
        test_set.shuffle()
        self.data_reader = BatchReader(test_set, ("Divergence", "Pressure"))

        with tf.variable_scope('model'):
            self.pred_pressure = predict_pressure(self.divergence_in)  # NN Pressure Guess

            # Pressure Solves with different Guesses (max iterations as placeholder)
            self.p_predGuess, self.iter_guess = solve_pressure(self.divergence_in, DOMAIN, pressure_solver=it_solver(self.max_it), guess=self.pred_pressure)
            self.p_trueGuess, self.iter_true = solve_pressure(self.divergence_in, DOMAIN, pressure_solver=it_solver(self.max_it), guess=self.true_pressure)
            self.p_noGuess, self.iter_zero = solve_pressure(self.divergence_in, DOMAIN, pressure_solver=it_solver(self.max_it), guess=None)

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

    #print mean iterations with predicted guess as average over validation batch
    def action_iterations(self):
        print("Mean Iterations (over entire test set) with current NN prediction as guess:")

        iterations_zero = []
        iterations_guess = []
        iterations_true = []

        #Look at each sample individually
        for i in range(100, 200):

            sample = self.data_reader[i]
            f_dict = {self.divergence_in.data: sample[0], self.true_pressure.data: sample[1], self.max_it: 500}

            #Solve for pressure for this sample using different guesses
            itZero, itPred, itTrue = self.session.run([self.iter_zero, self.iter_guess, self.iter_true], f_dict)

            iterations_zero.append(itZero)
            iterations_guess.append(itPred)
            iterations_true.append(itTrue)

            print(i)

        print("Zero Guess: ", np.mean(iterations_zero))
        print("Predicted Guess: ", np.mean(iterations_guess))
        print("True Guess: ", np.mean(iterations_true), "\n")

    #Use matplotlib to make diagram of residuum mean/max vs. iterations with and without guess
    def action_plot_iterations(self):
        residuum_max = []
        residuum_mean = []
        residuum_max_noguess = []
        residuum_mean_noguess = []

        all_maxima = []
        all_means = []
        all_maxima_noguess = []
        all_means_noguess = []
        it = []
        it_to_plot = 180
        batch_size = 60

        batch = self.data_reader[range(100, 100 + batch_size)]
        f_dict = {self.divergence_in.data: batch[0], self.true_pressure.data: batch[1]}

        self.info('Plot Residuum against Iterations...')

        for i in range(1, it_to_plot):
            f_dict[self.max_it] = i# set max_it to current i

            #solve for pressure with and without predicted guess
            div_in, pressure, pressure_noguess = self.session.run([self.divergence_in, self.p_predGuess, self.p_noGuess], f_dict)
            it.extend([i] * batch_size)

            #residuum with guess (absolute value)
            residuum = np.absolute((div_in.data - math.laplace(pressure.data))[:, 1:-1, 1:-1, :])
            batch_maxima = np.max(residuum, axis=(1,2,3))
            batch_means = np.mean(residuum, axis=(1,2,3))

            #record individual max/mean for scatterplot
            all_maxima.extend(batch_maxima)
            all_means.extend(batch_means)

            #record average mean/max over batch for curve plot
            residuum_max.append(np.mean(batch_maxima))
            residuum_mean.append(np.mean(batch_means))




            # residuum without guess (absolute value)
            residuum_noguess = np.absolute((div_in.data - math.laplace(pressure_noguess.data))[:, 1:-1, 1:-1, :])
            batch_maxima_noguess = np.max(residuum_noguess, axis=(1,2,3))
            batch_means_noguess = np.mean(residuum_noguess, axis=(1,2,3))

            #record individual max/mean for scatterplot
            all_maxima_noguess.extend(batch_maxima_noguess)
            all_means_noguess.extend(batch_means_noguess)

            residuum_max_noguess.append(np.mean(batch_maxima_noguess))
            residuum_mean_noguess.append(np.mean(batch_means_noguess))



        #Plot Maximum of Residuum
        plt.ylabel('Residuum Max (Blue: With Guess)')
        plt.yscale('log')
        plt.xlabel('Iterations')
        plt.plot(range(1, it_to_plot), residuum_max, 'b', range(1, it_to_plot), residuum_max_noguess, 'r')

        plt.scatter(x=it,y=all_maxima,s=0.01,c=(0,0,1.0),alpha=0.8)
        plt.scatter(x=it, y=all_maxima_noguess, s=0.01, c=(1.0, 0, 0), alpha=0.8)

        path = self.scene.subpath(name='residuumMax_vs_iterations')
        plt.savefig(path)
        plt.close()
        self.info('Saved Residuum Max Plot to %s' % path)

        #Plot Mean of Residuum
        plt.ylabel('Residuum Mean (Blue: With Guess)')
        plt.yscale('log')
        plt.xlabel('Iterations')
        plt.plot(range(1, it_to_plot), residuum_mean, 'b', range(1, it_to_plot), residuum_mean_noguess, 'r')

        plt.scatter(x=it,y=all_means,s=0.01,c=(0,0,1.0),alpha=0.8)
        plt.scatter(x=it, y=all_means_noguess, s=0.01, c=(1.0, 0, 0), alpha=0.8)

        path = self.scene.subpath(name='residuumMean_vs_iterations')
        plt.savefig(path)
        plt.close()
        self.info('Saved Residuum Mean Plot to %s' % path)

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
