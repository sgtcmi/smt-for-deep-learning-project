# smt-for-deep-learning-project

This is a project submission for the SMT4DL course done by Sougata Bhattacharya and Diganta
Mukhopadhyay.

The aim of this project is to develop a general framework to verify the invariance of DNNs under a
permutation of the inputs.

## Files:

- `direct_check.py` contains simple methods to directly check permutation invariance via a single
  z3 call.
- `perm_check.py` implements the main algorithm, which is partially implemented.
- `utils` contains several files with various utility functions, including functions to encode DNNs
  into z3, evaluate DNNs, and a function to time function calls.
- `examples` contains two examples currently, tic-tac-toe and periodfinder. Please see their
  respective readme files for details.

## Running the project:

Note that the main algo is implemented as a general callable function, that will be called by each
of the exapmles. Thus, to run the project, the examples themselves need to be run. The tic-tac-toe
example generates data, trains the network and then attempts to verify it. It uses both direct
checking, and the `perm_check` method. The data generation and network training is complete for the
period finder example. See the respective readmes for detail.
