"""
Use the algo to check for permutations
"""
from math import *
import time #PROFILE

import z3
import numpy as np

from utils import *



#def relu_vecs(vec):
#    """
#    Convert list of z3 expressions to a list representing the relu of those expressions.
#    """
#    return list(map(lambda z: z3.If(z >= 0, z, 0), vec))


def unit_vec(vec):
    """ 
    Convert a vector as a list of floats to a unit vector
    """
    size = sqrt(fsum(map(lambda x: x**2, vec)))
    return list(map(lambda x: x/size, vec))

def dot_prod(v1, v2):
    """
    Dot product of two vectors as lists of vectors
    """
    return fsum(( v1e*v2e for v1e, v2e in zip(v1, v2)))


def incl_check(vec, space):
    """ 
    Check if the given vector `vec` as a list of floats is in the space spanned by the vectors in
    `space`,which should be a list of unit vectors of the same dimensionality as `vec`
    """
    v = vec[:]
    for u in space:
        d = dot_prod(u, v)
        v = [ ve - ue*d for ve, ue in zip(v, u) ]
    for ve in v:
        if ve != 0:
            return False
    return True


def pull_back_relu(left_space, right_space):
    """
    Pull back `right_space` across relu to get a vecor in `left_space` that goes into
    `right_space` through the relu. Both spaces are given by a list of basis vecors, which are lists
    of floats all of the same length. Returns True, vec if found, False, [] otherwise.
    """
    assert(len(left_space[0]) == len(right_space[0]))

    relu_expr = lambda z: z3.If(z >= 0, z, 0)

    solver = z3.Solver()
    z3_rc = [ z3.Real('rc_%d'%i) for i in range(len(right_space)) ]     # vector in right_space
    z3_lc = [ z3.Real('lc_%d'%i) for i in range(len(left_space)) ]     # vector in right_space

    # constraints
    for i in range(len(left_space[0])):
        solver.add( z3.Sum([ rc*rb[i] for rc, rb in zip(z3_rc, right_space) ]) == 
                    relu_expr(z3.Sum([ lc*lb[i] for lc, lb in zip(z3_lc, left_space) ])))

    print('Calling solver to pull back across relu') #DEBUG
    if solver.check() == z3.unsat:
        return False, []
    else:
        mdl = solver.model()
        return True, [ mdl.eval(z3.Sum([ lc*lb[i] for lc, lb in zip(z3_lc, left_space) ]))
                            for i in range(len(left_space[0])) ]




def perm_check(weights, biases, perm, input_cond_hook):
    """
    Check if DNN given by the `weights` and `biases` is invariant under the given `perm`utation.
    `inp_conds_hook(inp_vars, solver)` asserts some input conditions on the z3 Reals `inp_vars` to
    the `solver` passed.
    """

    # Vars
    z3_vars = [ z3.Real('v_0_%d'%i) for i in range(len(perm)) ]
    #z3_vars_perm = [ z3.Real('p_v_0_%d'%i) for i in range(len(perm)) ]
    z3_nodes = [z3_vars]
    z3_nodes_perm = [[z3_vars[p] for p in perm]]
    z3_nodes_int = []
    z3_nodes_perm_int = []

    # Solver
    solver = z3.Solver()

    # Encode the fact that z3_vars and z3_vars_perm are inputs and permutatios of one another
    #solver.add(encode_perm_props.encode_perm(z3_vars, z3_vars_perm, perm))
    #input_cond_hook(z3_vars, solver) #TODO

    in_basis = []
    for i in range(len(perm)):
        b = [0]*(len(perm)*2)
        b[i] = 1
        b[perm[i] + len(perm)] = -1
        in_basis.append(b)
    

    for w, b, curr_lyr in zip(weights, biases, range(len(weights))):
        print('Kernel check for layer ', curr_lyr+1)
        l = len(w[0])
        out_basis = np.matrix(in_basis) * np.matrix([r + [0]*l for r in w] + 
                                                    [[0]*l + r for r in w])
        eq_basis  = out_basis           * np.matrix([[0]*i + [+1] + [0]*(l-i-1) for i in range(l)] + 
                                                    [[0]*i + [-1] + [0]*(l-i-1) for i in range(l)])
        if np.count_nonzero(eq_basis) == 0:
            print('Verified at layer via linear inclusion ', curr_lyr+1)
            return True, []
        else:
            print('Linear inclusion failed at layer ', curr_lyr+1)
            cex_basis = [ ib for ib, eb in zip(in_basis, out_basis) if np.count_nonzero(eb) > 0]
            for b in cex_basis:
                if not encode_dnn.eval_dnn(weights, biases, b[:len(w)]) == \
                        encode_dnn.eval_dnn(weights, biases, b[len(w):]):
                    print('Found cex at layer ', curr_lyr+1)
                    return False, b[:len(w)]
                print('Found no cex')

        

        return #DEBUG













#    return #DEBUG
#
#    relu_expr = lambda z: z3.If(z >= 0, z, 0)
#
#    # Layer by layer loop
#    curr_lyr = 0
#    for w, b in zip(weights, biases):
#        print('Exploring layer %d'%curr_lyr)
#        
#        z3_nodes_int.append([       z3.Real('v_i_%d_%d'%(curr_lyr, i))     for i in range(len(b)) ])
#        z3_nodes_perm_int.append([  z3.Real('p_v_i_%d_%d'%(curr_lyr, i))   for i in range(len(b)) ])
#        for i in range(len(b)):
#            solver.add( z3_nodes_int[curr_lyr][i] == ( z3.Sum([w[j][i]*z3_nodes[curr_lyr][j] 
#                                for j in range(len(w[i]))]) + b[i]) )
#            solver.add( z3_nodes_perm_int[curr_lyr][i] == ( z3.Sum([w[j][i]*z3_nodes_perm[curr_lyr][j] 
#                                for j in range(len(w[i]))]) + b[i]) )
#        
#        # Check if equal, else get check if cexs are true
#        solver.push()
#
#        solver.add( z3.Not( z3.And([ v == p_v for v,p_v in 
#                        zip(z3_nodes_int[curr_lyr], z3_nodes_perm_int[curr_lyr]) ])))
#        
#        #print(solver)
#        print('Calling solver')
#        p1 = time.process_time()
#        res = solver.check()
#        print('Time elapsed: ', time.process_time() - p1)
#        if res == z3.unsat:
#            print('Verified at layer %d'%curr_lyr)
#            return (True, [])
#        else:
#            mdl = solver.model()
#            cex = [ mdl.eval(z3_nodes[0][i]) for i in range(len(weights[0])) ]
#            cex_p = [ mdl.eval(z3_nodes_perm[0][i]) for i in range(len(weights[0])) ]
#            if not encode_dnn.eval_dnn(weights, biases, cex) == \
#                    encode_dnn.eval_dnn(weights, biases, cex_p):
#                print('CEX at layer %d'%curr_lyr)
#                #return (False, cex) #DEBUG no comment
#        solver.pop() 
#
#        curr_lyr += 1
#
#        # Add constraints for the next z3_nodes.
#        z3_nodes.append([       z3.Real('v_%d_%d'%(curr_lyr, i))    for i in range(len(b)) ])
#        z3_nodes_perm.append([  z3.Real('p_v_%d_%d'%(curr_lyr, i))  for i in range(len(b)) ])
#        for i in range(len(b)):
#            solver.add( z3_nodes[curr_lyr][i]       == relu_expr(z3_nodes_int[curr_lyr-1][i]) )
#            solver.add( z3_nodes_perm[curr_lyr][i]  == relu_expr(z3_nodes_perm_int[curr_lyr-1][i]) )
#
#        #return #DEBUG
#



# DEBUG
if __name__ == '__main__':

    from utils.misc import *
    n = 400
    id_basis = [ [0]*i + [1] + [0]*(n-i-1) for i in range(n) ]
    pb, vec = timeit(pull_back_relu, id_basis, id_basis)
    if pb:
        print('Pullback: ', vec)
    else:
        print('No pullback')
