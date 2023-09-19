"""rules - code for parsing and applying rules from CPA
"""

from difflib import SequenceMatcher
from rapidfuzz import process
import re

from cellprofiler_core.module import Module
from cellprofiler_core.setting import Binary

#from cellprofiler_core.setting.text import Float
FUZZY_FLOAT = 0.7 #We may eventually want to parametrize this with a setting, but let's not for now


import numpy



class Rules(Module):
    """Represents a set of CPA rules"""

    class Rule(object):
        """Represents a single rule"""

        def __init__(self, object_name, feature, comparitor, threshold, weights, allow_fuzzy, fuzzy_value):
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
            self.allow_fuzzy = allow_fuzzy
            self.fuzzy_value = fuzzy_value

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
                self.object_name, 
                self.return_fuzzy_measurement_name(
                    measurements.get_measurement_columns(),
                    self.object_name,
                    self.feature,
                    False,
                    self.allow_fuzzy
                    )
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

        @staticmethod
        def return_fuzzy_measurement_name(measurements,object_name,feature_name,full,allow_fuzzy,fuzzy_value=FUZZY_FLOAT):
            def standard_ratio(query, candidate, **kwargs):
                s = kwargs["SequenceMatcher"]
                s.set_seq1(candidate)
                return s.ratio()

            measurement_list = [f"{col[0]}_{col[1]}" for col in measurements]
            if allow_fuzzy:
                cutoff = fuzzy_value
            else:
                cutoff = 1

            query = '_'.join((object_name,feature_name))
            s = SequenceMatcher(b=query)
            closest_match = process.extractOne(
                query,
                measurement_list,
                processor=None,
                scorer=standard_ratio,
                score_cutoff=cutoff,
                scorer_kwargs={"SequenceMatcher": s}
            )

            if closest_match == None or len(closest_match) == 0:
                return ''
            else:
                if full:
                    return closest_match[0]
                else:
                    return closest_match[0][len(object_name)+1:] 



    def __init__(self,allow_fuzzy=False,fuzzy_value=FUZZY_FLOAT):
        """Create an empty set of rules.

        Use "parse" to read in the rules file or add rules programatically
        to self.rules.
        """
        self.rules = []
        self.allow_fuzzy = allow_fuzzy
        self.fuzzy_value = fuzzy_value

    def create_settings(self):

        self.allow_fuzzy = Binary(
            "Allow fuzzy feature matching?",
            False,
            doc="""
Allow CellProfiler to use the closest feature name, instead of only an exact match, when loading in 
Rules or a Classifier.

This may be necessary when long column names from the run where you generated the classification 
were truncated by ExportToDatabase.You can control this in ExportToDatabase in the "Maximum # of 
characters in a column name" setting. """
        )

        #possible future fuzzy_value setting goes here
    
    def settings(self):
        return [self.allow_fuzzy]

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
                        self.allow_fuzzy,
                        self.fuzzy_value
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
            rule = self.Rule(object_name, feature, ">", th, weights, self.allow_fuzzy,self.fuzzy_value)
            self.rules.append(rule)

            
