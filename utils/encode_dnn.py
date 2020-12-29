"""
Functions to encode a dnn into z3 expression
"""

import z3

def encode_network(weights, biases, z3_input):
    """
    Returns a z3 expression representing the output of the dnn when applied to the input represented
    by the list of z3 RealSort constants `z3_input`. The dnn is characterised by the given array of
    weights and biases, where `weights` is a list of weight matrices for each non-input layer, and
    `biases` is a list of corresponding bias vectors. The matrices are represented as lists of
    lists in a column first manner, vecorts as lists. Returns a list of z3 expressions representing
    the output.
    """

    # z3 consts for values of each layer
    z3_nodes = [z3_input]
    for w, b in zip(weights, biases):
        z3_nodes.append([z3.Sum([w[j][i]*z3_nodes[-1][j] for j in range(len(w[i]))])
                            + b[i] for i in range(len(b))])
        z3_nodes[-1] = [z3.If(z >= 0, z, 0) for z in z3_nodes[-1]]

    return z3_nodes[-1]


def assert_input_encoding(solver, z3_bool_input, z3_real_input):
    """
    Adds assertions to the given solver for the encoding relation between the BoolSort
    representation of the move and the RealSort representation of the input to the DNN
    """

    for b, r in zip(z3_bool_input, z3_real_input):
        solver.add(z3.If(b, r == 1, r == 0))


def eval_dnn(weights, biases, inp):
    """
    Evaluates a DNN on a given input
    """

    layer = inp
    for w, b in zip(weights, biases):
        layer_ = [ max(sum(( w[j][i]*layer[j] for j in range(len(w[i])) )) + b[i], 0) for i in range(len(b)) ]
        layer = layer_
    return layer
