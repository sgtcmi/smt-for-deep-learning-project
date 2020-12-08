"""
Use the algo to check for permutations
"""

import z3
from utils import *

import time #PROFILE
 
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


    relu_expr = lambda z: z3.If(z >= 0, z, 0)

    # Layer by layer loop
    curr_lyr = 0
    for w, b in zip(weights, biases):
        print('Exploring layer %d'%curr_lyr)
        
        z3_nodes_int.append([       z3.Real('v_i_%d_%d'%(curr_lyr, i))     for i in range(len(b)) ])
        z3_nodes_perm_int.append([  z3.Real('p_v_i_%d_%d'%(curr_lyr, i))   for i in range(len(b)) ])
        for i in range(len(b)):
            solver.add( z3_nodes_int[curr_lyr][i] == ( z3.Sum([w[j][i]*z3_nodes[curr_lyr-1][j] 
                                for j in range(len(w[i]))]) + b[i]) )
            solver.add( z3_nodes_perm_int[curr_lyr][i] == ( z3.Sum([w[j][i]*z3_nodes_perm[curr_lyr-1][j] 
                                for j in range(len(w[i]))]) + b[i]) )
        
        #z3_nodes.append([z3.Sum([w[j][i]*z3_nodes[-1][j] for j in range(len(w[i]))])
        #                    + b[i] for i in range(len(b))])
        ##z3_nodes[-1] = [z3.If(z >= 0, z, 0) for z in z3_nodes[-1]]
        #z3_nodes_perm.append([z3.Sum([w[j][i]*z3_nodes_perm[-1][j] for j in range(len(w[i]))])
        #                    + b[i] for i in range(len(b))])
        ##z3_nodes_perm[-1] = [z3.If(z >= 0, z, 0) for z in z3_nodes_perm[-1]]

        # Check if equal, else get check if cexs are true
        solver.push()

        solver.add( z3.Not( z3.And([ v == p_v for v,p_v in 
                        zip(z3_nodes_int[curr_lyr], z3_nodes_perm_int[curr_lyr]) ])))
        
        #print(solver)
        print('Calling solver')
        p1 = time.process_time()
        res = solver.check()
        print('Time elapsed: ', time.process_time() - p1)
        if res == z3.unsat:
            print('Verified at layer %d'%curr_lyr)
            return (True, [])
        else:
            mdl = solver.model()
            cex = [ mdl.eval(z3_nodes[0][i]) for i in range(len(weights[0])) ]
            cex_p = [ mdl.eval(z3_nodes_perm[0][i]) for i in range(len(weights[0])) ]
            if not encode_dnn.eval_dnn(weights, biases, cex) == \
                    encode_dnn.eval_dnn(weights, biases, cex_p):
                print('CEX at layer %d'%curr_lyr)
                return (False, cex)
        solver.pop() 

        curr_lyr += 1

        return #DEBUG



