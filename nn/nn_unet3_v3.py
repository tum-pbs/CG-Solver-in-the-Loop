# coding=utf-8
from nn_architecture import *
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

DOMAIN = Domain([64, 64], boundaries=CLOSED)  # [y, x]
DATAPATH = 'data/smoke_v3_highaccuracy/'  # has to match DOMAIN
DESCRIPTION = u"""
Train a neural network to predict the pressure corresponding to the given divergence field.
The predicted pressure should be able to be fed into a solver, reducing the iterations it needs to converge.

This version trains with normalized divergence fields and thus learns to output pressure fields that need to be
de-normalized by multiplying by the standard deviation of the original "Advected Velocity"'s divergence field.
"""


class SolverAssistedUnet(LearningApp):

    def __init__(self):
        LearningApp.__init__(self, 'Training', DESCRIPTION, learning_rate=2e-4, validation_batch_size=16, training_batch_size=32, base_dir="NN_UNet3_Basic", record_data=False)

        # --- placeholders ---
        self.divergence_in = divergence_in = placeholder(DOMAIN.centered_shape(batch_size=None))#Network Input
        self.true_pressure = true_pressure = placeholder(DOMAIN.centered_shape(batch_size=None))#Ground Truth


        # --- Build neural network ---
        with self.model_scope():
            self.pred_pressure = pred_pressure = predict_pressure(divergence_in)#NN Pressure Guess

            p_networkPlus10s, _ = solve_pressure(divergence_in, DOMAIN, pressure_solver=it_solver(10), guess=pred_pressure)
            p_networkPlus1s, _ = solve_pressure(divergence_in, DOMAIN, pressure_solver=it_solver(1), guess=pred_pressure)
            p_networkPlus2s, _ = solve_pressure(divergence_in, DOMAIN, pressure_solver=it_solver(2), guess=pred_pressure)
            p_networkPlus5s, _ = solve_pressure(divergence_in, DOMAIN, pressure_solver=it_solver(5), guess=pred_pressure)
            p_networkPlus15s, _ = solve_pressure(divergence_in, DOMAIN, pressure_solver=it_solver(15), guess=pred_pressure)

            # Pressure Solves with different Guesses (max iterations as placeholder)
            self.p_predGuess, self.iter_guess = solve_pressure(divergence_in, DOMAIN, pressure_solver=it_solver(500), guess=pred_pressure)
            self.p_trueGuess, self.iter_true  = solve_pressure(divergence_in, DOMAIN, pressure_solver=it_solver(500), guess=true_pressure)
            self.p_noGuess,   self.iter_zero  = solve_pressure(divergence_in, DOMAIN, pressure_solver=it_solver(500), guess=None)

        # --- Loss function ---
        loss = math.l2_loss(p_networkPlus15s - pred_pressure)
        self.add_objective(loss, 'Solver-Based Loss')

        # --- Dataset ---
        self.set_data(dict={self.divergence_in.data: 'Divergence', self.true_pressure.data: 'Pressure'},
                      train=Dataset.load(DATAPATH, range(0, 2799)),
                      val=Dataset.load(DATAPATH, range(2800, 2999)))

        # --- GUI ---
        self.add_field('Divergence', self.divergence_in)
        self.add_field('Predicted Pressure', pred_pressure)
        self.add_field('True Pressure', self.true_pressure)

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

    app = SolverAssistedUnet()
    app.prepare()
    app.info('Start Training CNN UNet3 to predict pressure from divergence (unsupervised)!')
    app.info('Loss: Solver-based')
    app.info('Train for %s steps...' % steps_to_train)

    def on_finish():
        app.info('Finished Training! (%s steps)' % steps_to_train)
        dir = app.scene.subpath('checkpoint_%08d' % app.steps)
        app.session.save(dir)

    app.play(max_steps=steps_to_train, callback=on_finish)

# run normally with GUI
else:
    app = show(display=('Predicted Pressure', 'True Pressure'))