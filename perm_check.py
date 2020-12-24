"""
Use the algo to check for permutations
"""
from math import *
import time #PROFILE

import z3
import numpy as np
import scipy.linalg as sp

from utils import *



#def relu_vecs(vec):
#    """
#    Convert list of z3 expressions to a list representing the relu of those expressions.
#    """
#    return list(map(lambda z: z3.If(z >= 0, z, 0), vec))

relu_expr = lambda z: z3.If(z >= 0, z, 0)

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
    `right_space` through the relu. Both affine spaces are given by a list of basis vecors of a
    linear space in a higher dimensional space, whose section where the last coordinate is 1 gives a
    vector in the affine subspace. Returns True, vec if found, False, [] otherwise. If returned, the
    last coordinate of the returned vec will be 1.
    """
    assert(len(left_space[0]) == len(right_space[0]))


    solver = z3.SolverFor("LRA")
    z3_rc = [ z3.Real('rc_%d'%i) for i in range(len(right_space)) ]     # vector in right_space
    z3_lc = [ z3.Real('lc_%d'%i) for i in range(len(left_space)) ]     # vector in right_space

    # constraints
    for i in range(len(left_space[0])):
        solver.add( z3.Sum([ rc*rb[i] for rc, rb in zip(z3_rc, right_space) ]) == 
                    relu_expr(z3.Sum([ lc*lb[i] for lc, lb in zip(z3_lc, left_space) ])))
    solver.add(z3_lc[-1] == 1)

    #TODO: Above takes a lot of time...?

    print('Calling solver to pull back across relu') #DEBUG
    if solver.check() == z3.unsat:
        return False, []
    else:
        mdl = solver.model()
        return True, [ mdl.eval(z3.Sum([ lc*lb[i] for lc, lb in zip(z3_lc, left_space) ]))
                            for i in range(len(left_space[0])) ]



def push_forward_relu(left_space):
    """
    Finds an affine subspace to the right of relu so that any vector in `left_space` affine space
    goes to it. Returns a basis of a larger space as a list of vecors. 
    """

    n = len(left_space[0])

    solver = z3.SolverFor("LRA")

    # Coefficients for the basis of the affine space
    z3_lcf = [ z3.Real('lcf_%d'%i) for i in range(n) ]

    # The vector on the left in the larger 
    z3_lvec = [ z3.Sum([ cf*lb[i] for cf, lb in zip(z3_lcf, left_space) ]) for i in range(n) ]
    solver.add(z3_lvec[-1] == 1)

    # The vector to the right
    z3_rvec = [ z3.Real('rv_%d'%i) for i in range(n) ]
    for lvc, rvc in zip(z3_lvec, z3_rvec):
        solver.add(rvc == relu_expr(lvc))

    right_basis = []
    z3_rcf = []

    for i in range(n):
        solver.push()
        print('Adding basis %d'%i, end='\r')

        # Linear independence conditions
        if len(right_basis) > 0:
            eq_expr = z3.Bool(True)
            for j in range(n):
                eq_expr = z3.And( eq_expr, z3_rvec[j] == z3.Sum([ cf * b[j] 
                                    for cf, b in zip(z3_rcf, right_basis) ]))
            solver.add(z3.Not(eq_expr))

        if solver.check() == z3.sat:
            mdl = solver.model()
            right_basis.append([ mdl.eval(rvc) for rvc in z3_rvec ])
            z3_rcf.append(z3.Real('rcf_%d'%len(z3_rcf)))
        else:
            break

        solver.pop()

    print('\nPushforward complete')

    return right_basis
        



def perm_check(weights, biases, perm):
    """
    Check if DNN given by the `weights` and `biases` is invariant under the given `perm`utation.
    """

    ## ABSTRACTION

    # Represent affine transforms as matrices in a larger space
    # Joint weight matrix
    dmat = [ np.matrix([r + [0]*len(w[0]) for r in w] + [[0]*len(w[0]) + r for r in w]) for w,b in zip(weights,biases) ]
    # Joint affine transform
    lmat = [ np.matrix([r + [0]*len(w[0]) + [0] for r in w] + [[0]*len(w[0]) + r + [0] for r in w] + [b + b + [1]])
                    for w,b in zip(weights,biases) ]
    # Kernels of joint weight matrix
    dkrn = [ np.transpose(sp.null_space(np.transpose(dm))) for dm in dmat]
    # Natrural inclusion of above kernels into higher space
    lkrn_p = [ [ r.tolist() + [0] for r in dk] for dk in dkrn]

    # Stats
    inp_dim = len(weights[0])
    out_dim = len(biases[-1])
    num_lyrs = len(weights)

    # Generate basis representing permutation constraint in the larger space
    in_basis = []
    for i in range(inp_dim):
        b = [0]*(inp_dim*2)
        b[i] = 1
        b[perm[i] + inp_dim] = -1
        in_basis.append(b + [0])
    in_basis.append((inp_dim*2)*[0] + [1])
    print(len(in_basis), len(in_basis[0]))
    
    # Track interpolants
    pre_lin_ints = []           # Interpolants
    post_lin_ints = []          # Interpolants after going through linear transform

    # Linear inclusion loop
    for w, b, lm, dm, curr_lyr in zip(weights, biases, lmat, dmat, range(num_lyrs)):
        print('Kernel check for layer ', curr_lyr+1)
        l = len(w[0])

        # Check linear inclusion by finding subbasis of input basis that does not go to 0. Represent
        # affine transform as a linear transform over a higher dimensional space
        # print(np.matrix(in_basis).shape, lm.shape) #DEBUG
        out_basis = np.matrix(in_basis) * lm
        eq_basis  = out_basis           * np.matrix([[0]*i + [+1] + [0]*(l-i-1) for i in range(l)] + 
                                                    [[0]*i + [-1] + [0]*(l-i-1) for i in range(l)] +
                                                    [[0]*l])

        if np.count_nonzero(eq_basis) == 0:
            print('Verified at layer via linear inclusion ', curr_lyr+1)
            #return True, [] #DEBUG
        else:
            print('Linear inclusion failed at layer ', curr_lyr+1)
            
            # This is the affine subspace from which we will pix cexs.
            cex_basis = [ ib for ib, eb in zip(in_basis, eq_basis) if np.count_nonzero(eb) > 0]

            # Pull back cex
            print('Attempting pullback....', end='')
            cex = cex_basis[0]
            for prel, pstl, idx in zip(reversed(pre_lin_ints), reversed(post_lin_ints),
                                        reversed(range(curr_lyr))):
                suc, cex = pull_back_relu(pstl, cex_basis)
                if not suc:
                    print('failed')
                    break
                
                # Pull back over affine transform using kernel
                cex = [ i-j for i,j in zip(cex[:-1], (b+b)) ]
                sl0, _, _, _ = sp.lstsq(np.transpose(dmat[idx]), np.asarray(cex))
                cex_basis = lkrn_p[idx] + [sl0 + [1]]

            print('success') 
            
            # Check if true cex
            if not encode_dnn.eval_dnn(weights, biases, cex[:inp_dim]) == \
                    encode_dnn.eval_dnn(weights, biases, cex[inp_dim:]):
                print('Found CEX')
                #return False, cex[:inp_dim] #DEBUG
            print('CEX is spurious')

        # Save these interpolants, and get next ones
        print('Looking for affine interpolant for next layer')
        pre_lin_ints.append(in_basis)
        out_basis = out_basis.tolist()
        post_lin_ints.append(out_basis)
        in_basis = push_forward_relu(out_basis)

    
    
    ## REFINEMENT

    # Set up solver and vars
    refined_solver = z3.SolverFor("LRA")
    # Stores the values going into each layer's transform in reverse order. First member stores
    # output.
    lyr_vars =  [[ z3.Real('lyr_%d_%d'%(num_lyrs, nn)) for nn in range(out_dim) ]]
    lyr_vars_p = [[ z3.Real('lyr_p_%d_%d'%(num_lyrs, nn)) for nn in range(out_dim) ]]
    for v, v_ in zip(lyr_vars[-1], lyr_vars_p[-1]):
        refined_solver.add(v == v_)

    # Refinement Loop
    for itp, w, b, i in zip(reversed(pre_lin_ints), reversed(weights), reversed(biases),
                            range(num_lyrs - 1, 0, -1)):
        print("Refining layer %d"%i)

        # Encode layers
        lyr_vars.append([ z3.Real('lyr_%d_%d'%(i, nn)) for nn in range(len(w)) ])
        lyr_vars_p.append([ z3.Real('lyr_p_%d_%d'%(i, nn)) for nn in range(len(w)) ])
        lyr_out = encode_dnn.encode_network([w], [b], lyr_vars[-1])
        lyr_out_p = encode_dnn.encode_network([w], [b], lyr_vars_p[-1])
        for le, lep, lv, lvp in zip(lyr_out, lyr_out_p, lyr_vars[-2], lyr_vars_p[-2]):
            refined_solver.add(z3.Not(le == lv))
            refined_solver.add(z3.Not(lep == lvp))

        # Perform refined check
        refined_solver.push()
        
        itp_vars = [ z3.Real('cf_%d'%i) for i in range(len(itp)) ]
        for i in range(len(w)):
            refined_solver.add( lyr_vars[-1][i] == z3.Sum([ iv * it[i] for iv, it in 
                                                            zip(itp_vars,itp) ]))
            refined_solver.add( lyr_vars_p[-1][i] == z3.Sum([ iv * it[len(w) + i] for iv, it in 
                                                            zip(itp_vars,itp) ]))
        refined_solver.add( 1 == z3.Sum([ iv * it[-1] for it in zip(itp_vars, itp) ]))

        if refined_solver.check() == z3.unsat:
            print("Verified via refinement at layer %d")
            #return True, [] #DEBUG
        else:
            print('Found potential cex, attempting pullback....', end='')

            mdl = refined_solver.model()
            # The cex is a single point in the joint affine space at the begining of the current
            # layer, and that can be represented by the follwing point
            cex_basis = [ list(map(mdl.eval, lyr_vars[-1])) + list(map(mdl.eval, lyr_vars_p[-1])) +
                            [1] ]
            
            # Pull back cex
            cex = cex_basis[0]
            for prel, pstl, idx in zip(reversed(pre_lin_ints[:i+1]), reversed(post_lin_ints[:i+1]),
                                        reversed(range(curr_lyr))):
                suc, cex = pull_back_relu(pstl, cex_basis)
                if not suc:
                    print('failed')
                    break
                
                # Pull back over affine transform using kernel
                cex = [ i-j for i,j in zip(cex[:-1], (b+b)) ]
                sl0, _, _, _ = sp.lstsq(np.transpose(dmat[idx]), np.asarray(cex))
                cex_basis = lkrn_p[idx] + [sl0 + [1]]

            print('success') 
            
            # Check if true cex, TODO: check if this should always return true, if so optimise
            if not encode_dnn.eval_dnn(weights, biases, cex[:inp_dim]) == \
                    encode_dnn.eval_dnn(weights, biases, cex[inp_dim:]):
                print('Found CEX')
                #return False, cex[:inp_dim] #DEBUG
            print('CEX is spurious')



        
    
            
    return #DEBUG
    #TODO complete
        




# DEBUG
if __name__ == '__main__':

    from utils.misc import *
    import random

    n = 100
    id_basis = [ [0]*i + [1] + [0]*(n-i-1) for i in range(n) ]
    vec = [ random.uniform(1, 100) for i in range(n) ]
    rand_sp = [ [ random.uniform(1, 100) for i in range(n) ] for j in range(n//2) ]
    pf = timeit(push_forward_relu, rand_sp)
    print(pf[:4])
