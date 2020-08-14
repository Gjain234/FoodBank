# Food Bank
This repository contains a reference implementation for the waterfilling algorithms for fair allocation of resources in online settings with only distribution data on demands.

### Dependencies
The code has been tested in `Python 3` and depends on a number of Python
packages.

* numpy
* matplotlib
* plotly
* pandas
* scipy

### Quick Tour

We offer implementations for online waterfilling algorithms that aim to maximize fairness a limited budget allocation.

The following files in `functions/` implement the different algorithms:
* `food_bank_functions.py`: implements all the different online waterfilling algorithms as well as an offline version
* `food_bank_bayesian.py`: implements the bayesian version of the waterfilling algorithm

These files found in `simulations/` run experiments with different demand distributions. We use a uniform, exponential, and simple distribution (equal probability of demand of 1 or 2) in our simulations. To run the experiments used in the paper, you should run `experiments_save_csv.py`.

Use the files in `figures/` to build figures for the saved data.

Each file has parameters at the top which can be changed in order to replicate the parameters considered for each experiment in the paper.
