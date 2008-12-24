""" tests - tests of CellProfiler-level modules

Also test numpy and scipy here
"""
__version = "$Revision: 1 $"

if __name__ == "__main__":
    import scipy.io.matlab
    import unittest
    import nose
    import sys

    scipy.io.matlab.test()
    nose.main()
    
