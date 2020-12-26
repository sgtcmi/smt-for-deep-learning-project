"""
Use the algo to check for permutations
"""
from math import *
import time #PROFILE
import itertools as itr

import z3
import numpy as np
import scipy.linalg as sp

from utils import *

"""
Some constants to tune some heuristics

pb_nbasis       -   During pullback of a counterexample, there is a potentially very large space
                   that needs to be searched. We reduce that by only taking pb-bsize many subsets
                   of the basis set during pullback across relu, and finding counterexamples for
                   each of these.
pb_nqueries     -   For each pullback across the relu, a sat query is made that can produce multiple
                    potential cexes. This parameter is the maximum number of cexes that we should
                    consider
"""
pb_nbasis = 2
pb_nqueries = 1


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




# New push forward
def push_forward_relu(left_space):
    """
    Finds an linear subspace to the right of relu so that any vector in `left_space` left space
    goes to it. Returns a basis of the space as a list of vecors. `left_space` must be a list of
    linearly indpenedent vectors giving a basis of the left space.
    """
    
    d = len(left_space)
    n = len(left_space[0])
    
    # Set up z3.
    solver = z3.SolverFor("LRA")
    z3_coeffs = [ z3.Real("c_%d"%i) for i in range(d) ]

    # Calculate the tie-classes of the relus
    tie_class_reps = [0]
    tie_class_vals = list(range(n))
    for i in range(1, n):
        print("Classifying ", i, end='\r', flush=True)    # DEBUG
        solver.push()
        solver.add( 0 >= z3.Sum([ c*l[i] for c, l in zip(z3_coeffs, left_space) ]) )
        cls = -1
        for rep in tie_class_reps:
            solver.push()
            solver.add( 0 < z3.Sum([ c*l[rep] for c, l in zip(z3_coeffs, left_space) ]) )
            if solver.check() == z3.unsat:
                cls = rep
                solver.pop()
                break
            solver.pop()
        if cls == -1:
            tie_class_reps.append(i)
        else:
            tie_class_vals[i] = cls
        solver.pop()
    print()


    fb_basis = []

    # If there are completely positive points in the left space, then those points exist in the
    # right space
    solver.reset()
    solver.push()
    for i in range(n):
        solver.add( 0 <= z3.Sum([ c*l[i] for c, l in zip(z3_coeffs, left_space) ]) )
    print("Checking completely positive points") #DEBUG
    if solver.check() == z3.sat:
        print("Completely positive points exist") #DEBUG
        fb_basis = left_space[:]
    solver.pop()

    # Now, for each tie-class, we add basis generating that tie class
    for rep, i in zip(tie_class_reps, range(len(tie_class_reps))):
        for b, j in zip(left_space, range(len(left_space))):
            print("Adding basis %d for tie-class %d"%(j, i), end='\r', flush=True)    #DEBUG
            b_ = [ c if rv == rep else 0 for c, rv in zip(b, tie_class_vals) ]
            if np.linalg.matrix_rank(np.asarray(fb_basis + [b_])) > len(fb_basis):
                fb_basis.append(b_)
                print("\nAdding basis")
    print()

    return fb_basis


class ReluPullbackIterator:
    """
    Iterates over all pullbacks of a space across a relu.
    """
    def __init__(self, right_space, is_affine):
        """
        Construct to pull back the `right_space`. Bool `is_affine` states if the given space represents
        an affine space, in which case the returned points will always have the last coordinate 1.
        """
        # Numbers
        self.r_space = right_space[:]
        self.d = len(right_space)
        self.n = len(right_space[0])
        self.r = min(self.d, pb_nbasis)
        self.k = factorial(self.n) // (factorial(self.r) * factorial(self.n-self.r))

        # Set up solver
        self.solver = z3.SolverFor("LRA")
        self.z3_rc = [ z3.Real('rc_%d'%i) for i in range(self.d) ]        # vector in right_space
        self.z3_lv = [ z3.Real('lc_%d'%i) for i in range(self.n) ]        # vector in left_space
        if is_affine:
            self.solver.add(self.z3_lv[-1] == 1)

        # Set up combination iterator
        self.subb_cmb = itr.combinations(zip(self.z3_rc, self.r_space), self.r)
        self.sbn = 0

        # Number of queries we have done. By this, we signal that we have not done any queries yet
        self.qn = pb_nqueries
        

    
    def __iter__(self):
        return self

    def __next__(self):
        
        while True:         
            # Repeat iteration until something is returned, or StopIteration is raised
            if self.qn >= pb_nqueries:
                # To next basis combination
                self.solver.reset()
                self.sbn += 1
                subb = next(self.subb_cmb) # This should raise StopIteration
                for i in range(self.n):
                    self.solver.add( z3.Sum([ rc*rb[i] for rc, rb in subb ]) == relu_expr(self.z3_lv[i]))
                self.qn = 0
            
            print("Calling solver with %d nodes and %d bases for combination %d of %d, query %d of %d"%( 
                    self.n, self.r, self.sbn, self.k, self.qn, pb_nqueries))
            # Continue doing queries
            if self.solver.check() == z3.sat:
                mdl = self.solver.model()
                rvals = [ mdl.eval(v) for v in self.z3_lv ]
                cex = [ rv.numerator_as_long() / rv.denominator_as_long() for rv in rvals ]
                self.solver.add( z3.Not( z3.And([ lv == c for lv, c in zip(self.z3_lv, cex) ])))
                self.qn += 1
                return cex
            else:
                self.qn = pb_nqueries  # Signal that we need to go to the next iteration

        

           

def pull_back_cex_explore(weights, biases, inp_dim,
                            jlt_mat, jlt_krn, 
                            pre_lin_ints, post_lin_ints, 
                            curr_lyr, cex_basis):
    """
    Given a basis of potential counterexamples at any layer, pulls the cex back over the network
    to the input layer and checks if the cex is true or spurious. The arguments are:

    weights, biases     -   Weights and biases of the network
    jlt_mat, jlt_krn    -   Matrices giving the joint linear transform, and their kernels for each
                            layer
    pre_lin_ints, 
    post_lin_ints       -   The interpolants derived for each layer, given as an affine space feeding into
                            the joint affine transform, and the corresponding affine space emerging
                            from the joint affine transform
    curr_lyr            -   The layer at which the potential counterexample basis was produced
    cex_basis           -   The potential counterexample basis

    """

    print('Attempting pullback at layer %d'%curr_lyr)   #DEBUG
    
    suc = True
    if curr_lyr > 0:
        idx = curr_lyr-1
        prel = pre_lin_ints[idx]
        pstl = post_lin_ints[idx]
        b = biases[idx]
        
        # iterate over pull backs across the relu in this layer
        for cex in ReluPullbackIterator(cex_basis, True):
        
            # Pull back over affine transform using kernel
            cex = [ i-j for i,j in zip(cex[:-1], (b+b)) ]
            npcex = np.asarray(cex)
            sl0, _, _, _ = sp.lstsq(np.transpose(jlt_mat[idx]), npcex)
            cex_basis = [ r + [0] for r in jlt_krn[idx].tolist() ] + [sl0.tolist() + [1]]

            # Pull back obtained basis in lower layers, return if something is found
            suc, cex = pull_back_cex_explore(weights, biases, inp_dim, jlt_mat, jlt_krn, pre_lin_ints,
                                                post_lin_ints, idx, cex_basis)
            if suc:
                return True, cex

        # Nothing is found for any pullback across relu
        return False, []
    else:

        print('Pullback reached layer 0')

        # Check if any basis is a true cex
        for cex in cex_basis:
            if encode_dnn.eval_dnn(weights, biases, cex[:inp_dim]) != \
                encode_dnn.eval_dnn(weights, biases, cex[inp_dim:]):
                print('Found CEX')
                return True, cex[:inp_dim]
        print('CEX is spurious')
        return False, [] 




def perm_check(weights, biases, perm):
    """
    Check if DNN given by the `weights` and `biases` is invariant under the given `perm`utation.
    """

    ## ABSTRACTION

    # Represent affine transforms as matrices in a larger space
    # Joint weight matrix
    jlt_mat = [ np.matrix([r + [0]*len(w[0]) for r in w] + [[0]*len(w[0]) + r for r in w]) for w,b in zip(weights,biases) ]
    # Joint affine transform
    jat_mat = [ np.matrix([r + [0]*len(w[0]) + [0] for r in w] + [[0]*len(w[0]) + r + [0] for r in w] + [b + b + [1]])
                    for w,b in zip(weights,biases) ]
    # Kernels of joint weight matrices
    jlt_krn = [ np.transpose(sp.null_space(np.transpose(dm))) for dm in jlt_mat]

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
    for w, b, lm, curr_lyr in zip(weights, biases, jat_mat, range(num_lyrs)):
        print('Kernel check for layer ', curr_lyr+1)
        l = len(w[0])

        # Check linear inclusion by finding subbasis of input basis that does not go to 0. Represent
        # affine transform as a linear transform over a higher dimensional space
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

            suc, cex = pull_back_cex_explore(weights, biases, inp_dim, jlt_mat, jlt_krn,
                                            pre_lin_ints, post_lin_ints, curr_lyr, cex_basis)

            if suc:
                #return False, cex #DEBUG
                pass

        # Save these interpolants, and get next ones
        print('Looking for affine interpolant for next layer')
        pre_lin_ints.append(in_basis)
        out_basis = out_basis.tolist()
        post_lin_ints.append(out_basis)
        in_basis = push_forward_relu(out_basis)     #TODO Confirm that out_basis is linearly ndependent


    
    
    ## REFINEMENT

    # Set up solver and vars
    refined_solver = z3.SolverFor("LRA")
    # Stores the values going into each layer's transform in reverse order. First member stores
    # output.
    lyr_vars =  [[ z3.Real('lyr_%d_%d'%(num_lyrs, nn)) for nn in range(out_dim) ]]
    lyr_vars_p = [[ z3.Real('lyr_p_%d_%d'%(num_lyrs, nn)) for nn in range(out_dim) ]]
    for v, v_ in zip(lyr_vars[-1], lyr_vars_p[-1]):
        refined_solver.add(z3.Not(v == v_))

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
            refined_solver.add(le == lv)
            refined_solver.add(lep == lvp)

        # Perform refined check
        refined_solver.push()
        
        # Cefficient to each basis in interpolant
        itp_cffs = [ z3.Real('cf_%d'%i) for i in range(len(itp)) ]
        # Encode that input is in inetrpolant, 
        for j in range(len(w)):
            refined_solver.add( lyr_vars[-1][j] == z3.Sum([ ic * it[j] for ic, it in 
                                                                    zip(itp_cffs, itp) ]))
            refined_solver.add( lyr_vars_p[-1][j] == z3.Sum([ ic * it[len(w) + j] for ic, it in 
                                                                    zip(itp_cffs, itp) ]))
        refined_solver.add( 1 == z3.Sum([ ic * it[-1] for ic, it in zip(itp_cffs, itp) ]))

        continue #DEBUG

        if refined_solver.check() == z3.unsat:
            print("Verified via refinement at layer %d")
            #return True, [] #DEBUG
        else:
            print('Found potential cex, attempting pullback....', end='')

            mdl = refined_solver.model()
            # The cex is a single point in the joint affine space at the begining of the current
            # layer, and that can be represented by the follwing line 
            cex_basis = [   list(map(mdl.eval, lyr_vars[-1])) + 
                            list(map(mdl.eval, lyr_vars_p[-1])) ]
            cex_basis[0] = [ v.numerator_as_long()/v.denominator_as_long() for v in cex_basis[0]]\
                            + [1]
            
            suc, cex = pull_back_cex_explore(weights, biases, inp_dim, jlt_mat, jlt_krn,
                                            pre_lin_ints, post_lin_ints, i, cex_basis)
            if suc:
                #return False, cex       #DEBUG
                pass



        
    
            
    return #DEBUG
    #TODO linear constraints on input
        




# DEBUG
if __name__ == '__main__':

    from utils.misc import *
    import random

    n = 500
    k = 3
    #id_basis = [ [0]*i + [1] + [0]*(n-i-1) for i in range(n) ]
    #vec = [ random.uniform(1, 100) for i in range(n) ]
    #rand_sp = [ [ random.uniform(1, 100) for i in range(n) ] for j in range(n//2) ]
    #sp1 = [[1, 0, 0], [0, 1, -1]]
    #pf = timeit(push_forward_relu, rand_sp)
    #for b in pf:
    #    for i in range(min(len(b), 10)):
    #        print("%6.3f, "%b[i], end='')
    #    print("..." if len(b) > 10 else "", flush=True)
    #print(push_forward_relu(sp1))

    rand_left =     [ [ random.uniform(1, 100) for i in range(n) ] for j in range(n-2) ]
    rand_right =    [ [ random.uniform(1, 100) for i in range(n) ] for j in range(n-1) ]
    rand_one =      [ [ random.uniform(1, 100) for i in range(n) ] for j in range(k)]
    pb = timeit(lambda x : [c for c in ReluPullbackIterator(x, True)], rand_right)
    print("Pullback done, returned %d cexes"%len(pb))
    
