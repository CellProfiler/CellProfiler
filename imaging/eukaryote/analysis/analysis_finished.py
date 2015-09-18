from __future__ import with_statement


class AnalysisFinished(object):
    def __init__(self, measurements, cancelled):
        self.measurements = measurements
        self.cancelled = cancelled
