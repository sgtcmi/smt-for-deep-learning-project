"""

Uses z3 to check for permutation invariance with one direct z3 call.

"""

import z3

from utils import *


def direct_check(weights, biases, perm, input_cond_hook):
    """
    Given a dnn via a list of `weights` and `biases`, and a permutation `perm`, check using a single
    z3 call if the DNN is invariant under the permutation of inputs. `perm` is a list where each
    element represent where that element goes to under the permutation. If invariance holds, returns
    True, else returns False and a counterexample. `input_cond_hook` is a function that enforces the
    constraints on the input by taking the z3 variables corresponding to the inputs, and a solver
    and adds constraints corresponding to the inputs to the solver, arguments: (z3_vars, solver)
    """

    z3_vars = [ z3.Real('v_%d'%i) for i in range(len(perm)) ]
    z3_vars_perm = [ z3.Real('p_v_%d'%i) for i in range(len(perm)) ]

    solver = z3.Solver()

    solver.add(encode_perm_props.encode_perm(z3_vars, z3_vars_perm, perm))
    solver.add( z3.Not( encode_dnn.encode_network(weights, biases, z3_vars) == 
                    encode_dnn.encode_network(weights, biases, z3_vars_perm)) )
    input_cond_hook(z3_vars, solver)
    
    if solver.check() == z3.unsat:
        return (True, [])
    else:
        mdl = solver.model()
        return (False, ([ mdl.eval(z3.Real('v_%d'%i)) for i in range(len(perm)) ], 
                        [ mdl.eval(z3.Real('p_v_%d'%i)) for i in range(len(perm)) ]))
