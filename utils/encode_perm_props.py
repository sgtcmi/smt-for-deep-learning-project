"""

This encodes the permutation invariance of a given NN as a z3 expression

"""

import z3

def encode_perm(z3_vars, z3_perm_vars, perm):
    """
    Encodes the fact that the the z3 variables given in the list `z3_vars` is a permutation of the
    variables in the list `z3_perm_vars` according to the permutation `perm`. Perm must be a list of
    the same size as that of `z3_vars` and `z3_perm_vars`, with each element being the position
    where that element is being sent by the permutation. Note that the expression returned is a
    conjunction of equalities.
    """

    return z3.And([ var == z3_perm_vars[i] for var, i in zip(z3_vars, perm) ])

