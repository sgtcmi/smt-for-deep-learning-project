"""
Loads and verifies a model that was trained by main.py. Looks for invariance under shifting the
signal by a given shift size. The properties are first attempted to be verified by the perm-check
algo, then it is given to z3.

Command line usage:
    python verify.py <model-file-name> <shift-size> <property-number> [property-specific-arguments]

The available properties are described below. In all cases, any aruments specified as the period
must not be more than the number of sample points.

    0   -   No preconditions, just looks for a random signal as a counterexample. Takes no
            arguments.
    1   -   Looks for signals with a specified period. Takes one argument - the period.
    2   -   Looks for a periodic signal with a given number of steps of equal size per repeating
            period, a step being a strech of time where the signal is constant. Takes two arguments,
            first being the size of the step, and the second being the number of steps. The period
            of the signal is calculated as the product of these 2 numbers. So, arguments 10, 4 would
            look for invariance of signals with 4 steps, and period 40.
    3   -   A constant signal. This should be verified. Takes no arguments.
    4   -   An arbitrary signal with period equal to the shift value. Takes no arguments. This is
            essentially the same as property 1 with the argument being the same as the shift size.
            This is again expected to be verified.
    5   -   A square wave of a given period. Takes one argument, the period. This must be even.
    6   -   A square wave with period being twice shift. Shift must be even. This will essentially
            flip the square wave.
"""

import sys
sys.path.insert(1, "../..")
from direct_check import *
from perm_check import *
from utils import *
import matplotlib.pyplot as plt
import time
import sys
import os
from tensorflow.keras import models, layers


def encode_period(inp_size, perd, lin_conds):
    """
    Adds to lin_conds equations stating that the period of the signal is perd. inp_size is the
    number of samples
    """
    m = inp_size//perd
    for i in range(perd):
        for j in range(m-1):
            eq = [ 0 for _ in range(inp_size + 1) ]
            eq[j*perd + i] = 1
            eq[(m-1)*perd + i] = -1
            lin_conds.append(eq)
    for i in range((m-1)*perd, inp_size):
        eq = [ 0 for _ in range(inp_size + 1) ]
        eq[i] = 1
        eq[i-perd] = -1


def encode_eq_strech(inp_size, start, stop, lin_conds):
    """
    Adds equations to lin_conds stating that the samples in range(start, stop) all have the same
    values
    """
    for j in range(start+1, stop):
        r = [ 0 for _ in range(inp_size + 1) ]
        r[start] = 1
        r[j] = -1
        lin_conds.append(r)

    

## Load model
try:
    model_file = sys.argv[1]
except:
    print("Bad command line usage")
    exit()

print("Loading model")
mdl = models.load_model(model_file)
weights = [l.get_weights()[0].tolist() for l in mdl.layers]
biases =  [l.get_weights()[1].tolist() for l in mdl.layers]
inp_size = len(weights[0])


## Generate permutation

try:
    shift = int(sys.argv[2])
except:
    print("Bad command line usage")
    exit()


perm = [ (i+shift)%inp_size for i in range(inp_size) ]



## Load linear preconditions

try:
    prcond = int(sys.argv[3])
except:
    print("Bad command line usage")
    exit()


lin_conds = []

if prcond == 1:
    print("Property 1 selected")
    try:
        perd = int(sys.argv[4])
    except:
        print("Bad command line usage")
        exit()
    encode_period(inp_size, perd, lin_conds)

elif prcond == 2:
    print("Property 2 selected")
    try:
        ssize = int(sys.argv[4])
        snum = int(sys.argv[5])
    except:
        print("Bad command line usage")
        exit()
    encode_period(inp_size, ssize*snum, lin_conds)
    for i in range(snum):
        encode_eq_strech(inp_size, i*ssize, (i+1)*ssize, lin_conds)

elif prcond == 3:
    print("Property 3 selected")
    encode_eq_strech(inp_size, 0, inp_size, lin_conds)

elif prcond == 4:
    print("Property 4 selected")
    encode_period(inp_size, shift, lin_conds)

elif prcond == 5:
    print("Property 5 selected")
    try:
        perd = int(sys.argv[4])
    except:
        print("Bad command line usage")
        exit()
    encode_period(inp_size, perd, lin_conds)
    encode_eq_strech(inp_size, 0, perd//2, lin_conds)
    encode_eq_strech(inp_size, perd//2, perd, lin_conds)

elif prcond == 6:
    print("Property 6 selected")
    encode_period(inp_size, shift*2, lin_conds)
    encode_eq_strech(inp_size, 0, shift, lin_conds)
    encode_eq_strech(inp_size, shift, 2*shift, lin_conds)

else:
    print("No preconditions added")

## Verification

print("Verifying, shift is: ", shift)
t0 = time.process_time()
res, cex = perm_check(weights, biases, perm, lin_conds)
print("Time: ", time.process_time() - t0)
if res:
    print('Verified')
else:
    print('Not verified') #, model: ', mdl)
    pcex = [ cex[p] for p in perm ]
    print('Output for model and permuted model:', mdl.predict([cex, pcex]))
    plt.plot([ i/inp_size for i in range(inp_size) ], cex, 'b', label='cex')
    plt.plot([ i/inp_size for i in range(inp_size) ], pcex, 'g', label='p(cex)')
    plt.show()


## Direct check for the above conditions
print("Setting up a direct check with z3")
t0 = time.process_time()
slv = z3.SolverFor("LRA")
inps = [ z3.Real("z_%d"%i) for i in range(inp_size) ]
inps_p = [ z3.Real("z_p_%d"%i) for i in range(inp_size) ]
print("Initalized variables, time: ", t0 - time.process_time() )

# Encode perm
for i in range(inp_size):
    print("Encoding permutation for variable %d, time: "%i, time.process_time() - t0, end='\r')
    slv.add( inps[i] == inps_p[perm[i]] )
print()

# Encode linear conditions 
for eq, i in zip(lin_conds, range(len(lin_conds))):
    print("Encoding precondition linear equation %d, time: "%i, time.process_time() - t0, end='\r' )
    slv.add( z3.Sum([ cf*ip for cf, ip in zip(eq[:-1], inps) ]) == eq[-1] )
print()

# Encode non invariance
print("Encoding network, time: ", time.process_time() - t0 )
dval = encode_dnn.encode_network(weights, biases, inps)
dval_p = encode_dnn.encode_network(weights, biases, inps_p)
andexp = dval[0] == dval_p[0]
for i in range(1, len(dval)):
    print("Encoding permutation invariance for output %d of %d, time:"%(i, len(dval)),
            time.process_time() - t0)
    andexp = z3.And(andexp, dval[i] == dval_p[i])
slv.add( z3.Not(andexp))
print("Encoding done, time: ", time.process_time() - t0 )

print("Calling z3")
t0 = time.process_time()
res = slv.check()
print("Time: ", time.process_time() - t0, res)
if res == z3.unsat:
    print('Verified')
else:
    z3_mdl = slv.model()
    z3_cex = [ z3_mdl.eval(ip) for ip in inps ]
    cex = [ zv.numerator_as_long() / zv.denominator_as_long() for zv in z3_cex ]
    print('Not verified')
    pcex = [ cex[p] for p in perm ]
    print('Output for model and permuted model:', mdl.predict([cex, pcex]))
    plt.plot([ i/inp_size for i in range(inp_size) ], cex, 'b', label='cex')
    plt.plot([ i/inp_size for i in range(inp_size) ], pcex, 'g', label='p(cex)')
    plt.show()

