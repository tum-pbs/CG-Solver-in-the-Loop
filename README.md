# Solver-Based Learning of Pressure Fields for Eulerian Fluid Simulation

In this work, we investigated different approaches to training a Convolutional Neural Network (CNN)
so that it can infer an approximate pressure solution that can be used as an initial guess for a 
Conjugate Gradient (CG) solver in a Eulerian fluid simulation.

In particular, we compared three different loss formulations:
1. SUP: A standard supervised loss
2. PHY: An unsupervised, physics-informed loss (directly minimizing the residual divergence after correcting with the pressure guess)
3. SOL: A loss including the differentiable CG solver which minimizes the difference of the network's prediction to a limited-iteration CG solution computed on top of the network's guess

This repository contains the source code used to generate the data, train the models and analyze their performance.
It also includes the snapshot of Φ<sub>*Flow*</sub> that was used for these experiments, which is not the most current version.

## Test Dataset Performance
Comparing the three approaches on sample divergence fields from our test dataset, we observed that the physics-based loss leads to a higher accuracy of the network's prediction itself, yet performs much worse as an initial guess for the CG solver than the solver-based approach in particular.

![TestsetResidual](documentation/figures/results_testset_residual.PNG)

Given an initial guess by the model trained with the solver-based loss, the CG solver's residual divergence decreases very quickly over the first few iterations. This leads to the solver needing much fewer iterations to reach target accuracies like 10^-3 than it would from a zero guess or that of the physics-informed model.

![TestsetIterations](documentation/figures/results_testset_iterations.PNG)

## Simulation Performance
We evaluated the pressure predictors' in-simulation performance when used to fully replace the CG solver (Network simulation) and when used as an in conjunction with the CG solver to provide an initial guess (hybrid simulation).

### Network Simulation
The Physics-based model proved much more stable than either of the others, which was expected as its output has the highest standalone accuracy (as shown on the test dataset). However, the solver-based model showed much more unstable behaviour than anticipated. This instability can be traced to a checkerboard pattern that our solver-based approach introduces into its prediction. These artifacts lead to accumulating residual error as the network makes consecutive predicitons. 

We suspect this pattern to be the result of training the network with a direct difference to a CG-solver iteration with limited iterations. In the initial training steps, the network's guess is similar to a zero guess. Therefore, minimizing its difference to a CG solution computed by performing e.g. 5 iterations on top of it makes the network emulate the checkerboard patterns these intermediate CG solutions often contain.

### Hybrid Simulation
Using the trained models together with the CG solver, the solver-based approach fares much better. As on the test dataset, it is the most effective approach at reducing the iterations the solver needs to reach its target accuracy. This leads to tangible simulation speed-ups. Using a hybrid simulation scheme and our SOL model, simulation steps with an accuracy target of 10^-3 require less than half the computation time compared to a standard simulation step with zero guess.

![HybridSimPerformance](documentation/figures/results_hybridsim_performance.PNG)

## Combining Solver-Based and Physics-Informed Training
To leverage both the stability advantage we observed for the physics-informed model and the reduction in solver iterations provided by the solver-based training approach, we combined them into a fourth loss formulation. This approach no longer contains a direct difference of the network's output to an intermediate solver solution. Instead, it uses the output plus 5 additional differentiable solver iterations to correct the input divergence. The combined loss then minimizes the residual divergence of that correction.

We found that this effectively eliminated the checkerboard pattern and stability issues our previous solver-based approach showed, while maintaining the iteration reduction it enabled. The combined loss formulation can therefore be used to train models that function well as standalone pressure solvers and as providers of an initial pressure guess for the CG solver.

![CombinedLoss](documentation/figures/results_testset_combinedloss.PNG)
