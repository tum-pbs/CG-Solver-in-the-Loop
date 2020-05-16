# coding=utf-8
from nn_architecture import *
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

DOMAIN = Domain([64, 64], boundaries=CLOSED)  # [y, x]
DATAPATH = 'data/smoke_v3_test/'  # has to match DOMAIN
DESCRIPTION = u"""
Train a neural network to predict the pressure corresponding to the given divergence field.
The predicted pressure should be able to be fed into a solver, reducing the iterations it needs to converge.

This version recreates the Tompson paper approach by using the predicted pressure to correct the velocity,
then calculating its divergence as the loss term.
"""

def correct(velocity, pressure, domain):

    # set up fluiddomain with boundary conditions
    active_mask = math.ones(domain.centered_shape(name='active', extrapolation='constant'))
    accessible_mask = active_mask.copied_with(extrapolation=Material.accessible_extrapolation_mode(domain.boundaries))

    fluiddomain = FluidDomain(domain, active=active_mask, accessible=accessible_mask)

    # apply boundary conditions
    gradp = StaggeredGrid.gradient(pressure)
    gradp = fluiddomain.with_hard_boundary_conditions(gradp)

    # correct and return
    return (velocity - gradp)


class TompsonUnet(LearningApp):

    def __init__(self):
        LearningApp.__init__(self, 'Training', DESCRIPTION, learning_rate=2e-4, validation_batch_size=16, training_batch_size=32, base_dir="NN_UNet3_Basic", record_data=False)

        # --- placeholders ---
        # Centered Grids
        self.divergence_in = divergence_in = placeholder(DOMAIN.centered_shape(batch_size=None))  # Network Input
        self.true_pressure = true_pressure = placeholder(DOMAIN.centered_shape(batch_size=None))  # Ground Truth

        # create a geometry mask that is 0: at the border and 1: everywhere else
        g_mask = math.ones(DOMAIN.centered_shape())
        g_mask.data[:, 0, :, :] = 0
        g_mask.data[:, :, 0, :] = 0
        g_mask.data[:, :, -1, :] = 0
        g_mask.data[:, -1, :, :] = 0

        # Staggered Grids
        staggered_shape = (None, *(DOMAIN.resolution + 1), DOMAIN.rank)

        v_in_data = tf.placeholder(dtype=tf.float32, shape=staggered_shape)
        self.v_in = StaggeredGrid(v_in_data)  # Velocity corresponding to Network Input (Divergence)

        v_true_data = tf.placeholder(dtype=tf.float32, shape=staggered_shape)
        self.v_true = StaggeredGrid(v_true_data)  # Velocity corrected, Ground Truth


        # --- Build neural network ---
        with self.model_scope():
            self.pred_pressure = pred_pressure = predict_pressure(divergence_in, g_mask.data)  # NN Pressure Guess

            #Tompson loss quantities (û)
            self.v_corrected = correct(self.v_in, DOMAIN.centered_grid(pred_pressure), DOMAIN)
            self.v_corrected_true = correct(self.v_in, self.true_pressure, DOMAIN)


            # Pressure Solves with different Guesses (max iterations as placeholder)
            self.p_predGuess, self.iter_guess = solve_pressure(divergence_in, DOMAIN, pressure_solver=it_solver(500), guess=pred_pressure)
            self.p_trueGuess, self.iter_true  = solve_pressure(divergence_in, DOMAIN, pressure_solver=it_solver(500), guess=true_pressure)
            self.p_noGuess,   self.iter_zero  = solve_pressure(divergence_in, DOMAIN, pressure_solver=it_solver(500), guess=None)

        # --- Tompson Loss function ---
        residuum = self.v_corrected.divergence(physical_units=False)
        div_loss = math.l2_loss(residuum)

        self.add_objective(div_loss, 'Tompson Loss')

        # --- Dataset ---
        self.set_data(dict={self.divergence_in.data: 'Divergence', self.true_pressure.data: 'Pressure', v_in_data: 'Divergent Velocity', v_true_data: 'Corrected Velocity'},
                      train=Dataset.load(DATAPATH, range(0, 2799)),
                      val=Dataset.load(DATAPATH, range(2800, 2999)))

        # --- GUI ---
        self.add_field('Divergence', self.divergence_in)
        self.add_field('Predicted Pressure', pred_pressure)
        self.add_field('True Pressure', self.true_pressure)
        self.add_field('Advected Velocity', self.v_in)
        self.add_field('Advected Velocity (Divergence)', self.v_in.divergence(physical_units=False))
        self.add_field('Corrected Velocity (NN)', self.v_corrected)
        self.add_field('Corrected Velocity (True)', self.v_true)
        self.add_field('Corrected Velocity (with True Pressure)', self.v_corrected_true)
        self.add_field('Residuum', residuum)

        self.add_field('Geometry Mask', g_mask)

        self.save_path = EditableString("Save/Load Path", self.scene.subpath('checkpoint_%08d' % self.steps))

        # --- TensorBoard ---
        self.add_scalar("Iterations (Predicted Guess)", self.iter_guess)
        self.add_scalar("Iterations (True Guess)", self.iter_true)
        self.add_scalar("Iterations (Zero Guess)", self.iter_zero)

    def action_save_model(self):
        self.session.save(self.save_path)

    def action_load_model(self):
        self.load_model(self.save_path)



# run from command line with fixed amount of steps
if len(sys.argv) > 1:
    steps_to_train = int(sys.argv[1])

    app = TompsonUnet()
    app.prepare()
    app.info('Start Training CNN UNet3 to predict pressure from divergence (unsupervised)!')
    app.info('Loss: Tompson Approach (Divergence of NN-Corrected Velocity)')
    app.info('Train for %s steps...' % steps_to_train)

    def on_finish():
        app.info('Finished Training! (%s steps)' % steps_to_train)
        dir = app.scene.subpath('checkpoint_%08d' % app.steps)
        app.session.save(dir)

    app.play(max_steps=steps_to_train, callback=on_finish)

# run normally with GUI
else:
    app = show(display=('Predicted Pressure', 'True Pressure'))