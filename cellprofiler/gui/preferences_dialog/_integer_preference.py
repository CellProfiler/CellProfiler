class IntegerPreference(object):
    """User interface info for an integer preference

    This signals that a preference should be displayed and edited as
    an integer, optionally limited by a range.
    """

    def __init__(self, minval=None, maxval=None):
        self.minval = minval
        self.maxval = maxval
