"""
Functions associated with legacy python
"""

try:
    # Python 2
    _cmp = cmp

    def cmp(a, b):
        return _cmp(a, b)

except NameError:
    # Python 3
    def cmp(a, b):
        return (a > b) - (a < b)
