# coding=utf-8
from phi.tf.flow import *
from phi.math import upsample2x

import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DOMAIN = Domain([64, 64], boundaries=CLOSED)  # [y, x]
DATAPATH = 'data/smoke_v2/'  # has to match DOMAIN
DESCRIPTION = u"""
Train a neural network to predict the pressure corresponding to the given divergence field.
The predicted pressure should be able to be fed into a solver, reducing the iterations it needs to converge.

This basic version does not support obstacles or closed borders.
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


# Predict pressure using Neural Network
def predict_pressure(divergence):
    result = pressure_unet(divergence.data)

    return CenteredGrid(result, divergence.box, name='pressure')


class TrainingTest(LearningApp):

    def __init__(self):
        LearningApp.__init__(self, 'Training', DESCRIPTION, learning_rate=2e-4, validation_batch_size=16, training_batch_size=32, base_dir="NN_UNet3_Basic", record_data=False)

        # --- placeholders ---
        self.divergence_in = divergence_in = placeholder(DOMAIN.centered_shape(batch_size=None))#Network Input
        self.true_pressure = true_pressure = placeholder(DOMAIN.centered_shape(batch_size=None))#Ground Truth

        self.max_it = self.editable_int("Max_Iterations", 500, (500,500))# Only used for manual plotting

        # --- Build neural network ---
        with self.model_scope():
            self.pred_pressure = pred_pressure = predict_pressure(divergence_in)#NN Pressure Guess

            p_networkPlus10s, _ = solve_pressure(divergence_in, DOMAIN, pressure_solver=it_solver(10), guess=pred_pressure)
            p_Zero10s, _        = solve_pressure(divergence_in, DOMAIN, pressure_solver=it_solver(10), guess=None)

            # Pressure Solves with different Guesses (max iterations as placeholder)
            self.p_predGuess, self.iter_guess = solve_pressure(divergence_in, DOMAIN, pressure_solver=it_solver(self.max_it), guess=pred_pressure)
            self.p_trueGuess, self.iter_true  = solve_pressure(divergence_in, DOMAIN, pressure_solver=it_solver(self.max_it), guess=true_pressure)
            self.p_noGuess,   self.iter_zero  = solve_pressure(divergence_in, DOMAIN, pressure_solver=it_solver(self.max_it), guess=None)

        # --- Loss function ---
        loss = math.l2_loss(p_networkPlus10s - pred_pressure)
        self.add_objective(loss, 'Loss')

        # --- Dataset ---
        self.set_data(dict={self.divergence_in.data: 'Divergence', self.true_pressure.data: 'Pressure'},
                      train=Dataset.load(DATAPATH, range(0, 279)),
                      val=Dataset.load(DATAPATH, range(280, 299)))

        # --- GUI ---
        self.add_field('Divergence', self.divergence_in)
        self.add_field('Predicted Pressure', pred_pressure)
        self.add_field('True Pressure', self.true_pressure)

        self.save_path = EditableString("Save/Load Path", self.scene.subpath('checkpoint_%08d' % self.steps))

        # --- TensorBoard ---
        self.add_scalar("Iterations (Predicted Guess)", self.iter_guess)
        self.add_scalar("Iterations (True Guess)", self.iter_true)
        self.add_scalar("Iterations (Zero Guess)", self.iter_zero)

    #print mean iterations with predicted guess as average over validation batch
    def action_iterations(self):
        print("Mean Iterations (over validation batch) with current NN prediction as guess:")

        iterations_zero = []
        iterations_guess = []
        iterations_true = []

        #Look at each sample individually
        for i in range(self.validation_batch_size):

            sample = self._val_reader[i]
            f_dict = self._feed_dict(sample, False)

            #Solve for pressure for this sample using different guesses
            itZero, itPred, itTrue = self.session.run([self.iter_zero, self.iter_guess, self.iter_true], f_dict)

            iterations_zero.append(itZero)
            iterations_guess.append(itPred)
            iterations_true.append(itTrue)

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
        it_to_plot = 140

        batch = self._val_reader[0:self.validation_batch_size]
        f_dict = self._feed_dict(batch, False)

        self.info('Plot Residuum against Iterations...')

        for i in range(1, it_to_plot):
            f_dict[self.max_it] = i# set max_it to current i

            #solve for pressure with and without predicted guess
            div_in, pressure, pressure_noguess = self.session.run([self.divergence_in, self.p_predGuess, self.p_noGuess], f_dict)
            it.extend([i] * self.validation_batch_size)

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

    def action_save_model(self):
        self.session.save(self.save_path)

    def action_load_model(self):
        self.load_model(self.save_path)



# run from command line with fixed amount of steps
if len(sys.argv) > 1:
    steps_to_train = int(sys.argv[1])

    app = TrainingTest()
    app.prepare()
    app.info('Start Training CNN UNet3 to predict pressure from divergence (unsupervised)!')
    app.info('Train for %s steps...' % steps_to_train)

    def on_finish():
        app.info('Finished Training! (%s steps)' % steps_to_train)
        dir = app.scene.subpath('checkpoint_%08d' % app.steps)
        app.session.save(dir)
        app.action_plot_iterations()

    app.play(max_steps=steps_to_train, callback=on_finish)

# run normally with GUI
else:
    app = show(display=('Predicted Pressure', 'True Pressure'))