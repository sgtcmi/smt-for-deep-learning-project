"""
Use the algo to check for permutations
"""
from copy import deepcopy
from math import *
import random
import time #PROFILE
import itertools as itr

import z3
import numpy as np
import scipy.linalg as sp

from utils import *

"""
Some constants to tune some heuristics

pb_nbasis_r,    -   During pullback of a counterexample, there is a potentially very large space
pb_nbasis_l         that needs to be searched. We reduce that by only taking pb-bsize many subsets
                    of the basis set during pullback across relu, and finding counterexamples for
                    each of these, for each of the left and right spaces.
pb_nqueries     -   For each pullback across the relu, a sat query is made that can produce multiple
                    potential cexes. This parameter is the maximum number of cexes that we should
                    consider
"""
pb_nbasis_r = 2 #2
pb_nbasis_l = 2 #2
pb_ntclass = 1
pb_nqueries = 1
pb_nrandq = 1000
pb_rand_lb = 0
pb_rand_ub = 1
fp_abs_tol = 1e-12


relu_expr = lambda z: z3.If(z >= 0, z, 0)

def check_cex(weights, biases, perm, prc_eq, cex):
    """
    Check if given cex violates invariance of given network under given permutation. Returns True if
    it does, False otherwise
    """
    assert len(weights[0]) == len(perm) and len(cex) == len(perm)
    for eq in prc_eq:
        if not isclose(sum([ c*v for c, v in zip(cex, eq[:-1]) ]), eq[-1], abs_tol=fp_abs_tol):
            print("CEX does not respect linear preconditions", 
                    sum([ c*v for c, v in zip(cex, eq[:-1]) ]) - eq[-1]) #DEBUG
            return False
    for x, y in zip(encode_dnn.eval_dnn(weights, biases, cex),
            encode_dnn.eval_dnn(weights, biases, [ cex[p] for p in perm ])):
        if not isclose(x, y, abs_tol=fp_abs_tol):
            return True
    return False


def space_intersect(asp, bsp):
    """
    Given two spaces as the span of two basis sets, find the intersection space as the span of the
    basis sets. 
    """
    assert(len(asp[0]) == len(bsp[0]))

    ssp = asp + [ [ -c for c in r ] for r in bsp ]
    k = np.transpose(sp.null_space(np.transpose(np.matrix(ssp))))
    if len(k) <= 0:
        return []
    return (np.matrix([ r[:len(asp)] for r in k ]) @ asp).tolist()

    
def extend_basis(b, vs):
    """
    Extends basis with list of vectors vs, checks linear dependence
    """
    b = deepcopy(b)
    for v in vs:
        if np.linalg.matrix_rank(np.asarray(b + [v])) > len(b):
            b.append(v)
    return b


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
    h_cp = False
    for i in range(n):
        solver.add( 0 <= z3.Sum([ c*l[i] for c, l in zip(z3_coeffs, left_space) ]) )
    print("Checking completely positive points") #DEBUG
    if solver.check() == z3.sat:
        print("Completely positive points exist") #DEBUG
        fb_basis = deepcopy(left_space)
        h_cp = True 
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

    return fb_basis, tie_class_reps, tie_class_vals, h_cp


        
class ReluPullbackIterator:
    """
    Given a space to the right of the relu, this iterator creates progressively more refined spaces
    to the left of the relu that go inside the right space under the relu.
    """
    def __init__(self, right_space, tie_class_reprs, tie_class_vals, has_cp):
        """
        Construct to pull back the `right_space`, refining iteratively. 
        """
        # Fields
        self.h_cp = has_cp
        self.r_space = deepcopy(right_space)
        self.tc_vals = deepcopy(tie_class_vals)
        self.rd = len(right_space)
        self.n = len(right_space[0])
        self.r = min(self.rd, pb_ntclass)
        self.k = (factorial(self.rd) // (factorial(self.r) * factorial(self.rd-self.r))) 

        # Keep track of which combinations of tie classes have been searched
        self.cmb = itr.combinations(tie_class_reprs, self.r)
        self.ncmb = 0

        # Number of queries we have done. By this, we signal that we have not done any queries yet
        self.qn = pb_nqueries

        # Tracks the set of bases returned previously
        self.prev_ret = []
        

    
    def __iter__(self):
        return self

    def __next__(self):
        
        while True:
            if self.h_cp:
                self.h_cp = False
                self.prev_ret = self.r_space
                print("Returning the right space, unrefined.")  
                return self.r_space

            else:
                bz = []
                bnz = []
                clss = next(self.cmb)
                self.ncmb += 1
                for i in range(self.n):
                    if self.tc_vals[i] in clss:
                        bz.append([0]*i + [1] + [0]*(self.n-i-1))
                    else:
                        bnz.append([0]*i + [1] + [0]*(self.n-i-1))
                b1 = space_intersect(bnz, self.r_space)
                if len(b1) <= 0:
                    continue
                self.prev_ret = extend_basis(self.prev_ret, extend_basis(b1, bz))
                print("Returning space refined with %d of %d tie-class combinations"%(self.ncmb,
                    self.k))
                return self.prev_ret
                
           

def pull_back_cex_explore(weights, biases, inp_dim, 
                            perm, prc_eq, in_basis,
                            tc_reps, tc_vals, has_cp,
                            curr_lyr, spr_basis):
    """
    Given a basis of non-counterexamples at any layer, uses a pullback procedure to search for
    counterexamples.

    weights, biases     -   Weights and biases of the network
    inp_dim             -   Dimensions of the input to the nn
    perm                -   Permutation to validate final cex against
    prc_eq              -   The precondition equations on the input
    in_basis            -   The basis of the space of allowed inputs
    curr_lyr            -   The layer at which the potential counterexample basis was produced
    spr_basis           -   The space known not to yield any counterexamples

    """

    print('Attempting pullback at layer %d'%curr_lyr)   #DEBUG
    
    if curr_lyr > 0:
        idx = curr_lyr-1
        aj = jat_mat[idx].tolist()
        
        # iterate over pull backs across the relu in this layer
        for gsp in ReluPullbackIterator(spr_basis, tc_reps[idx], tc_vals[idx], has_cp[idx]):
        
            # Pull back over affine transform
            bs = aj + [ [ -v for v in r ] for r in gsp ]
            n_spr_basis = [ r[:len(aj)] for r in 
                                    np.transpose(sp.null_space(np.transpose(bs))).tolist() ]

            # Pull back obtained basis in lower layers, return if something is found
            suc, cex = pull_back_cex_explore(weights, biases, inp_dim, perm, prc_eq, in_basis, 
                                                tc_reps, tc_vals, has_cp, idx, n_spr_basis)
            if suc:
                return True, cex

        # Nothing is found for any pullback across relu
        return False, []
    else:

        print('Pullback reached layer 0')

        # Decompose input space
        b_space = space_intersect(in_basis, spr_basis)
        if len(b_space) <= 0:
            g_space = in_basis
        else:
            g_space = space_intersect(in_basis, np.transpose(sp.null_space(np.matrix(b_space))))
        assert(len(g_space) > 0)

        # Give up if all last elements of bases are 0
        allz = True
        for r in g_space:
            allz = allz and isclose(r[-1], 0, abs_tol=fp_abs_tol)
        for r in b_space:
            allz = allz and isclose(r[-1], 0, abs_tol=fp_abs_tol)
        if allz:
            print("All last components are 0")
            return False, []

        # Now, try random combinations
        for tr in range(pb_nrandq):
            print("Trying random combination %d of %d"%(tr, pb_nrandq), end='\r')

            # Nonzero contribution from g_space
            gcf = [0] * len(g_space)
            allz = True
            for i in range(len(g_space)):
                gcf[i] = random.uniform(pb_rand_lb, pb_rand_ub)
                allz = allz and isclose(gcf[i], 0, abs_tol=fp_abs_tol)

            # Adjust if contribution from g_space is 0
            while allz:
                idx = random.randint(0, len(g_space))
                gcf[idx] = random.uniform(pb_rand_lb, pb_rand_ub)
                if not isclose(gcf[idx], 0, abs_tol=fp_abs_tol):
                    allz = False

            # Any contribution from b_space
            bcf = [ random.uniform(pb_rand_lb, pb_rand_ub) for _ in b_space ]

            # Sum of last elements, used to get a point from a vector in affine space
            lst = sum([ gc*gb[-1] for gc, gb in zip(gcf, g_space) ]) + \
                    sum([ bc*bb[-1] for bc, bb in zip(bcf, b_space) ])

            # Adjust lst if 0
            if isclose(lst, 0, abs_tol=fp_abs_tol):
                for i in range(len(gcf)):
                    if not isclose(g_space[i][-1], 0, abs_tol=fp_abs_tol):
                        lst += g_space[i][-1]
                        gcf[i] += 1
                for i in range(len(bcf)):
                    if not isclose(b_space[i][-1], 0, abs_tol=fp_abs_tol):
                        lst += b_space[i][-1]
                        bcf[i] += 1

            # Potential counterexample point
            cx = [ (sum([ gc*gb[i] for gc, gb in zip(gcf, g_space) ]) + 
                    sum([ bc*bb[i] for bc, bb in zip(bcf, b_space) ])) / lst for i in range(inp_dim) ]
            
            if check_cex(weights, biases, perm, prc_eq, cx):
                print("\nFound CEX")
                return True, cx
        
        print("\nNo random combination worked, backtracking cex search")
        return False, []


class RefinedReluPullbackIterator:
    """
    Pulls back a given space to the right of the relu to a point in the left of the relu in a given
    left space. Iterates through each possible such pullback point.
    """
    def __init__(self, left_space, right_space, is_affine):
        """
        Construct to pull back the `right_space`. Bool `is_affine` states if the given space represents
        an affine space, in which case the returned points will always have the last coordinate 1.
        """
        assert(len(left_space[0]) == len(right_space[0]))

        # Fields
        self.affine = is_affine
        self.r_space = right_space[:]
        self.l_space = left_space[:]
        self.rd = len(right_space)
        self.ld = len(left_space)
        self.n = len(right_space[0])
        self.rr = min(self.rd, pb_nbasis_r)
        self.lr = min(self.ld, pb_nbasis_l)
        self.k = (factorial(self.rd) // (factorial(self.rr) * factorial(self.rd-self.rr))) * \
                 (factorial(self.ld) // (factorial(self.lr) * factorial(self.ld-self.lr)))

        # Set up solver
        self.solver = z3.SolverFor("LRA")
        self.z3_rc = [ z3.Real('rc_%d'%i) for i in range(self.rd) ]        # Coefficients in right_space
        self.z3_lc = [ z3.Real('lc_%d'%i) for i in range(self.ld) ]        # Coefficients in left

        # Set up combination iterator
        self.subb_cmb = itr.product(itr.combinations(zip(self.z3_lc, self.l_space), self.lr),
                                    itr.combinations(zip(self.z3_rc, self.r_space), self.rr))
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
                print("Resetting, too many queries")
                self.solver.reset()
                self.sbn += 1
                subb_l, subb_r = next(self.subb_cmb)
                
                if self.affine:
                    self.solver.add(z3.Sum([ lc*lb[-1] for lc, lb in subb_l ]) == 1)
                for i in range(self.n):
                    self.solver.add( z3.Sum([ rc*rb[i] for rc, rb in subb_r ]) == 
                                    relu_expr(z3.Sum([ lc*lb[i] for lc, lb in subb_l ])))
                self.qn = 0
            
            print("Calling solver with %d nodes and %d bases for combination %d of %d, query %d of %d"%( 
                    self.n, self.rr, self.sbn, self.k, self.qn, pb_nqueries))
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


def refined_pull_back_cex_explore(weights, biases, inp_dim, in_space,
                            perm, prc_eq, jlt_mat, jlt_krn, post_lin_ints, curr_lyr, cex_basis):
    """
    Given a basis of potential counterexamples at any layer, pulls the cex back over the network
    to the input layer and checks if the cex is true or spurious. The arguments are:

    weights, biases     -   Weights and biases of the network
    inp_dim             -   Dimensions of the input to the nn
    perm                -   Permutation to validate final cex against
    prc_eq              -   The precondition equations on the input
    jlt_mat, jlt_krn    -   Matrices giving the joint linear transform, and their kernels for each
                            layer
    post_lin_ints,      -   The interpolants derived for each layer, given as an affine space feeding into
    pre_lin_ints,           the joint affine transform, and the corresponding affine space emerging
                            from the joint affine transform
    curr_lyr            -   The layer at which the potential counterexample basis was produced
    cex_basis           -   The potential counterexample basis

    """

    print('Attempting pullback at layer %d'%curr_lyr)   #DEBUG
    
    suc = True
    if curr_lyr > 0:
        idx = curr_lyr-1
        pstl = post_lin_ints[idx]
        b = biases[idx]
        
        # iterate over pull backs across the relu in this layer
        for cex in RefinedReluPullbackIterator(pstl, cex_basis, True):
        
            # Pull back over affine transform using kernel
            cex = [ i-j for i,j in zip(cex[:-1], (b+b)) ]
            npcex = np.asarray(cex)
            sl0, _, _, _ = sp.lstsq(np.transpose(jlt_mat[idx]), npcex)
            cex_basis = [ r + [0] for r in jlt_krn[idx].tolist() ] + [sl0.tolist() + [1]]

            # Pull back obtained basis in lower layers, return if something is found
            suc, cex = pull_back_cex_explore(weights, biases, inp_dim, in_space, perm, prc_eq,
                                                jlt_mat, jlt_krn, pre_lin_ints,
                                                post_lin_ints, idx, cex_basis)
            if suc:
                return True, cex

        # Nothing is found for any pullback across relu
        return False, []
    else:

        print('Pullback reached layer 0')
        if len(prc_eq) > 0:

            # DEBUG
            # Try and intersect cex_basis with in_space and pick random from there
            cx_b_trunc = [ r[:inp_dim] + [r[-1]] for r in cex_basis ]
            in_b_trunc = [ r[:inp_dim] + [r[-1]] for r in in_basis ]
            pcex_b = space_intersect(cx_b_trunc, prc_eq)
            if len(pcex_b) <= 0:
                return False, []


        # Give up if all last elements of g_space are 0
            allz = True
            for r in g_space:
                allz = allz and (r[-1] == 0)
            if allz:
                print("All last components are 0")
                return False, []

            # First try bases
            for b in pcex_b:
                if b[-1] != 0 and check_cex(weights, biases, perm, prc_eq, 
                                [ b[i] / b[-1] for i in range(inp_dim) ]):
                    print("Lucky basis")
                    return True, [ b[i] / b[-1] for i in range(inp_dim) ]

            # Now, try random combinations
            for tr in range(pb_nrandq):
                print("Trying random combination %d of %d"%(tr, pb_nrandq), end='\r')

                # Generate coefficients
                cf = [0] * len(pcex_b)
                allz = True
                for i in range(len(pcex_b)):
                    cf[i] = random.uniform(pb_rand_lb, pb_rand_ub)
                    allz = allz and (gcf[i] == 0)

                # Adjust if contribution is 0
                while allz:
                    idx = random.randint(0, len(pcex_b))
                    cf[idx] = random.uniform(pb_rand_lb, pb_rand_ub)
                    if cf[idx] != 0:
                        allz = False

                # Sum of last elements, used to get a point from a vector in affine space
                lst = sum([ c*b[-1] for c, b in zip(cf, pcex_b) ])

                # Adjust lst if 0
                if lst == 0:
                    for i in range(len(cf)):
                        if pcex_b[i][-1] != 0:
                            lst += pcex_b[i][-1]
                            cf[i] += 1

                # Potential counterexample point
                cx = [ sum([ c*b[i] for c, b in zip(cf, pcex_b) ]) / lst for i in range(inp_dim) ]
                
                if check_cex(weights, biases, perm, prc_eq, cx):
                    print("\nFound CEX")
                    return True, cx
            
        else:
            # Check if any basis is a true cex
            for cex in cex_basis:
                if check_cex(weights, biases, perm, prc_eq, cex[:inp_dim]):
                    print('Found CEX')
                    return True, cex[:inp_dim]
        print('CEX is spurious')
        return False, [] 





def perm_check(weights, biases, perm, prc_eq):
    """
    Check if DNN given by the `weights` and `biases` is invariant under the given `perm`utation. The
    other arguments are:
    prc_eq          -   A set of linear equalities giving a precondition on the input. Specified as
                        a list of rows, each row [a1, a2, ... an, b] specifies the equation a1.x1 +
                        a2.x2 + ... an.xn = b.
    """

    # Numbers
    inp_dim = len(weights[0])
    out_dim = len(biases[-1])
    num_lyrs = len(weights)

    # Generate basis representing permutation constraint in the larger space
    in_basis = []
    if(len(prc_eq) > 0):
        # Convert the constraints into a basis for a precondition space
        prc_a = np.matrix([ r[:-1] for r in prc_eq])
        prc_b = np.asarray([r[-1] for r in prc_eq])
        prc_eq_krn = np.transpose(sp.null_space(prc_a)).tolist()
        prc_eq_s0, _, _, _ = sp.lstsq(prc_a, prc_b)
        prc_eq_s0 = prc_eq_s0.tolist()
        
        if len(prc_eq_krn) <= 0:
            # The equations determine exactly one input, and so we just check if that is a cex
            if check_cex(weights, biases, perm, prc_eq, prc_eq_s0):
                return False, prc_eq_s0
            return True, []
        
        in_basis = [ r + [r[p] for p in perm] + [0] for r in prc_eq_krn] + \
                    [ prc_eq_s0 + [prc_eq_s0[p] for p in perm] + [1] ]
    else:
        for i in range(inp_dim):
            b = [0]*(inp_dim*2)
            b[i] = 1
            b[perm[i] + inp_dim] = -1
            in_basis.append(b + [0])
        in_basis.append((inp_dim*2)*[0] + [1])
    prc_space = [ r[:] for r in in_basis ]


    

    ## ABSTRACTION


    # Represent affine transforms as matrices in a larger space
     # Joint weight matrix
    jlt_mat = [ np.matrix([r + [0]*len(w[0]) for r in w] + [[0]*len(w[0]) + r for r in w]) 
                                            for w,b in zip(weights,biases) ]
    # Joint affine transform
    jat_mat = [ np.matrix([r + [0]*len(w[0]) + [0] for r in w] + 
                        [[0]*len(w[0]) + r + [0] for r in w] + [b + b + [1]])
                                            for w,b in zip(weights,biases) ]
    # Kernels of joint weight matrices
    jlt_krn = [ np.transpose(sp.null_space(np.transpose(dm))) for dm in jlt_mat]


    # Track interpolants
    pre_lin_ints = []           # Interpolants
    post_lin_ints = []          # Interpolants after going through linear transform
    tc_reps = []                # Representatives of tie classes
    tc_vals = []                # Values of each tie class
    has_cp = []                 # Weather the push forward of the interpolant has the complete space
                                # of the previous layer

    # Linear inclusion loop
    for w, b, lm, curr_lyr in zip(weights, biases, jat_mat, range(num_lyrs)):
        print('Kernel check for layer ', curr_lyr+1)
        l = len(w[0])

        # Find image of in_basis under space. Ensure generated out_basis is linearly independent
        out_basis = []
        for b in in_basis:
            ob = (np.array(b) @ lm).tolist()[0]
            _r = np.linalg.matrix_rank(np.asarray(out_basis + [ob]))
            if _r > len(out_basis):
                out_basis.append(ob)

        # Check linear inclusion by finding subbasis of input basis that does not go to 0
        eq_basis = lm @ np.matrix([[0]*i + [+1] + [0]*(l-i-1) for i in range(l)] + 
                                  [[0]*i + [-1] + [0]*(l-i-1) for i in range(l)] +
                                  [[0]*l])

        if np.allclose(np.matrix(in_basis) @ eq_basis, 0):
            print('Verified at layer via linear inclusion ', curr_lyr+1)
            return True, [] 
        else:
            print('Linear inclusion failed at layer ', curr_lyr+1)

            # The basis containing the space at this layer where no CEX can ever be found
            spr_basis = np.transpose(sp.null_space(np.transpose(eq_basis))).tolist()

            # Backtracking pullback for cex
            suc, cex = pull_back_cex_explore(weights, biases, inp_dim, perm, prc_eq, prc_space,
                                            tc_reps, tc_vals, has_cp, curr_lyr, spr_basis)

            if suc:
                return False, cex 
                pass

        # Save these interpolants, and get next ones
        print('Looking for affine interpolant for next layer')
        pre_lin_ints.append(in_basis)
        post_lin_ints.append(out_basis)
        in_basis, tc_r, tc_v, h_cp = push_forward_relu(out_basis) 
        tc_reps.append(tc_r)
        tc_vals.append(tc_v)
        has_cp.append(h_cp)


    
    
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
                            reversed(range(num_lyrs))):
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


        if refined_solver.check() == z3.unsat:
            print("Verified via refinement at layer %d")
            return True, []
        else:
            print('Found potential cex, attempting pullback....', end='')

            mdl = refined_solver.model()
            # The cex is a single point in the joint affine space at the begining of the current
            # layer, and that can be represented by the follwing line 
            cex_basis = [   list(map(mdl.eval, lyr_vars[-1])) + 
                            list(map(mdl.eval, lyr_vars_p[-1])) ]
            cex_basis[0] = [ v.numerator_as_long()/v.denominator_as_long() for v in cex_basis[0]]\
                            + [1]
            
            suc, cex = refined_pull_back_cex_explore(weights, biases, inp_dim, prc_space, perm, prc_eq,
                                            jlt_mat, jlt_krn, post_lin_ints, i, cex_basis)
            if suc:
                return False, cex
    
            
    assert(False) #We should never reach here
        




   
