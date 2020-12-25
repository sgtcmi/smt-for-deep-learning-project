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

class TimePoint:
    """
    Stores time at creation, and at each point time_point() is called, shows time elapsed since
    creation and since last time time_point() was called.
    """

    def __init__(self, nm=""):
        self.t0 = time.process_time()
        self.tl = time.process_time()
        self.nam = nm
        print("Timing %s"%nm)


    def time_point(self, msg):
        t = time.process_time()
        print("%s: %s, Time %.4f, Deltime %.4f"%(self.nam, msg, t - self.t0, t - self.tl))
        self.tl = 1

