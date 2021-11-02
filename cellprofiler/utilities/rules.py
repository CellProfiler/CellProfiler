"""rules - code for parsing and applying rules from CPA
"""

import re

import numpy


class Rules(object):
    """Represents a set of CPA rules"""

    class Rule(object):
        """Represents a single rule"""

        def __init__(self, object_name, feature, comparitor, threshold, weights):
            """Create a rule

            object_name - the name of the object in the measurements
            feature - the name of the measurement (for instance,
                      "AreaShape_Area")
            comparitor - the comparison to be performed (for instance, ">")
            threshold - the positive / negative threshold for the comparison
            weights - a 2xN matrix of weights where weights[0,:] are the
                      scores for the rule being true and
                      weights[1,:] are the scores for the rule being false
                      and there are N categories.
            """
            self.object_name = object_name
            self.comparitor = comparitor
            self.threshold = threshold
            self.feature = feature
            self.weights = weights

        def score(self, measurements):
            """Score a rule

            measurements - a measurements structure
                           (cellprofiler_core.measurements.Measurements). Look
                           up this rule's measurement in the structure to
                           get the testing value.
            Returns a MxN matrix where M is the number of measurements taken
            for the given feature in the current image set and N is the
            number of categories as indicated by the weights.
            """
            values = measurements.get_current_measurement(
                self.object_name, self.feature
            )
            if values is None:
                values = numpy.array([numpy.NaN])
            elif numpy.isscalar(values):
                values = numpy.array([values])
            score = numpy.zeros((len(values), self.weights.shape[1]), float)
            if len(values) == 0:
                return score
            mask = ~(numpy.isnan(values) | numpy.isinf(values))
            if self.comparitor == "<":
                hits = values[mask] < self.threshold
            elif self.comparitor == "<=":
                hits = values[mask] <= self.threshold
            elif self.comparitor == ">":
                hits = values[mask] > self.threshold
            elif self.comparitor == ">=":
                hits = values[mask] >= self.threshold
            else:
                raise NotImplementedError('Unknown comparitor, "%s".' % self.comparitor)
            score[mask, :] = self.weights[1 - hits.astype(int), :]
            score[~mask, :] = self.weights[numpy.newaxis, 1]
            return score

    def __init__(self):
        """Create an empty set of rules.

        Use "parse" to read in the rules file or add rules programatically
        to self.rules.
        """
        self.rules = []

    def parse(self, fd_or_file):
        """Parse a rules file

        fd_or_file - either a filename or a file descriptor. Parse assumes
                     that fd_or_file is a file name if it's a string or
                     unicode, otherwise it assumes that it's a file descriptor.
        """
        line_pattern = (
            "^IF\\s+\\((?P<object_name>[^_]+)"
            "_(?P<feature>\\S+)"
            "\\s*(?P<comparitor>[><]=?)"
            "\\s*(?P<threshold>[^,]+)"
            ",\\s*\\[\\s*(?P<true>[^\\]]+)\\s*\\]"
            ",\\s*\\[\\s*(?P<false>[^\\]]+)\\s*\\]\\s*\\)$"
        )
        if isinstance(fd_or_file, str):
            fd = open(fd_or_file, "r")
            needs_close = True
        else:
            fd = fd_or_file
            needs_close = False
        try:
            for line in fd:
                line = line.strip()
                match = re.match(line_pattern, line)
                if match is not None:
                    d = match.groupdict()
                    weights = numpy.array(
                        [
                            [float(w.strip()) for w in d[key].split(",")]
                            for key in ("true", "false")
                        ]
                    )
                    rule = self.Rule(
                        d["object_name"],
                        d["feature"],
                        d["comparitor"],
                        float(d["threshold"]),
                        weights,
                    )
                    self.rules.append(rule)
            if len(self.rules) == 0:
                raise ValueError("No rules found in %s" % str(fd_or_file))
        finally:
            if needs_close:
                fd.close()

    def score(self, measurements):
        """Score the measurements according to the rules list"""
        if len(self.rules) == 0:
            raise ValueError("No rules to apply")
        score = self.rules[0].score(measurements)
        for rule in self.rules[1:]:
            partial_score = rule.score(measurements)
            if partial_score.shape[0] > score.shape[0]:
                temp = score
                score = partial_score
                partial_score = temp
            score_len = partial_score.shape[0]
            score[:score_len, :] += partial_score[:score_len, :]
            if score.shape[0] > partial_score.shape[0]:
                score[score_len:, :] = numpy.NAN
        return score

    def get_classes(self):
        if len(self.rules) == 0:
            return []
        return [str(i + 1) for i in range(self.rules[0].weights.shape[0])]

    def get_features(self):
        return [rule.feature for rule in self.rules]

    def load(self, rules):
        # Convert a FastGentleBoosting classifier object into a set of rules
        for name, th, pos, neg, _ in rules:
            object_name, feature = name.split('_', 1)
            weights = numpy.vstack((pos, neg))
            rule = self.Rule(object_name, feature, ">", th, weights)
            self.rules.append(rule)
