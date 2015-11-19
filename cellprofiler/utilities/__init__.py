'''Utilities for CellProfiler
'''


########################
#
# Futures for itertools
#
########################

import sys as _sys
if _sys.version_info[0] > 2 or _sys.version_info[1] >= 6:
    from itertools import product
else:
    def product(*args):
        '''The cartesian product of the arguments
        
        see docs for itertools.product for full documentation
        (available in Python 2.6+)
        '''
        lengths = [len(arg) for arg in args]
        total = reduce(lambda a,b: a*b, lengths)
        for idx in range(total):
            t = total
            result = []
            i1 = idx
            for arg,length in zip(args,lengths):
                t = t / length
                result.append(arg[int(i1 / t)])
                i1 = i1 % t
            yield result
