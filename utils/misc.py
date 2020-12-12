"""
Misc utils
"""

import time

def timeit(f, *args, name=''):
    """
    Time f using given arguments and return it's value, print time in fractional seconds to stdout
    """

    p1 = time.process_time()
    ret = f(*args)
    p2 = time.process_time()
    print('Time for function %s %f'%(name, p2 - p1))
    return ret
