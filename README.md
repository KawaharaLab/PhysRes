# PHysRes
**P**hysical **Hys**teretic **R**eservoir

## Setup
This baseline code is based on our paper:

>C. Caremel, Y. Kawahara, K. Nakajima, Hysteretic reservoir, Physical Review Applied 22 (6), 064045, 2024.

To test for a simple example, simply run the script `predictNARMA10.py` in your preferred IDE (e.g. Spyder, Jupyter notebook).\
A plot of the prediction should be displayed, along with the NRMSE error relative to the target (expected error < 0.1).

## Model
Hysteretic behavior can be defined as a dynamic shift, or lag, denoted $\lambda$, between the input and the output of a system. 
The hysteretic encoding $\sigma$ is defined as returning 0 if $k < \lambda$, where $k$ denotes the k-th element of the input vector to this operation. Otherwise, the output of some activation function (generally a nonlinear function, such as the sigmoid function, with scaling hyperparameter $\alpha$) is returned.

The hysteretic reservoir model size is set with $N$ units, some initial state condition $x_0$, and driven with external input $u(t)$, where each state update at time step $t$ is defined by the functional output of the hysteretic encoding described previously, and dependent on the previous state (here we fix $\tau=1$):\
${x}(t) = \sigma(u(t) \cdot W_{in} + {x}(t-\tau))$, where $W_{in}$ represents a fixed input weight matrix (random numbers drawn from the uniform distribution).

In code, the model is inititalized with `physres = prc.PHysRes(u, N, x0, lamda, alpha, tau)`.

## Task example

The input timeseries is the NARMA10 task, where $u_k$ and $A_k$ represent the pair of inputs (also random numbers drawn from the uniform distribution) and the corresponding target:\
$A_{k+1} = 0.3A_k + 0.05A_k(\sum_{i=0}^9A_{k-i}) + 1.5u_{k-9}u_k + 0.1$

The training of the output weights $W_{out}$ is then done via linear regression over the target $y$, and called via `physres.Run()`:\
$W_{out} = W_{out}^{+} \cdot y$

The prediction $y_p$ is then done over the trained state $x_t$:\
$y_p = x_t \cdot W_{out}$

This can be obtained from `physres.Test(0.7, Y)` where the parameter 0.7 represents the ratio of the number of training samples over the total number of points in the timeseries, and Y is the target timeseries.

## Hyperparameters
The class is instantiated via prc.PHysRes(u, N, x0, lamda, alpha, tau) where u represents the input, N is the number of nodes in the network, x0 is the initial condition, lamda is the latency hyperparameter, alpha is the scaling hyperparameter and tau is the state delay in the reservoir.
Although it is recommended to keep the defaults settings, latency and scaling are the two main hyperparameters. One can run a grid search over those for fine-tuning.
Normalization should be done with min-max for best results, but sine wave is used in the physical implementation.
Also, the error can be computed with either NRMSE (recommended) or NMSE.
For more details, please check: https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.22.064045

## Credits
If you use any ideas from the papers or code in this repo, please consider citing the authors:

```bibtex
@article{PhysRevApplied.22.064045,
  title = {Hysteretic reservoir},
  author = {Caremel, Cedric and Kawahara, Yoshihiro and Nakajima, Kohei},
  journal = {Phys. Rev. Appl.},
  volume = {22},
  issue = {6},
  pages = {064045},
  numpages = {15},
  year = {2024},
  month = {Dec},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevApplied.22.064045},
  url = {https://link.aps.org/doi/10.1103/PhysRevApplied.22.064045}
}
```
