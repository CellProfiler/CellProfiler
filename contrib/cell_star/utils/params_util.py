# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

# External imports
import numpy as np
from contrib.cell_star.config.config import default_config
import sys


def default_parameters(segmentation_precision=-1, avg_cell_diameter=-1, min_size=0, max_size=sys.maxint):
    parameters = default_config()
    if avg_cell_diameter != -1:
        parameters["segmentation"]["avgCellDiameter"] = avg_cell_diameter

    if segmentation_precision == -1:
        return parameters
    else:
        return parameters_from_segmentation_precision(parameters, segmentation_precision)


def parameters_from_segmentation_precision(parameters, segmentation_precision):
    if "use_exact" in parameters:
        # make parameters that are int ints
        parameters["segmentation"]["seeding"]["from"]["cellContentRemovingCurrSegmentsRandom"] = int(parameters["segmentation"]["seeding"]["from"]["cellContentRemovingCurrSegmentsRandom"])
        parameters["segmentation"]["seeding"]["from"]["snakesCentroidsRandom"] = int(parameters["segmentation"]["seeding"]["from"]["snakesCentroidsRandom"])
        parameters["segmentation"]["seeding"]["from"]["cellBorderRemovingCurrSegmentsRandom"] = int(parameters["segmentation"]["seeding"]["from"]["cellBorderRemovingCurrSegmentsRandom"])
        parameters["segmentation"]["seeding"]["from"]["cellBorderRandom"] = int(parameters["segmentation"]["seeding"]["from"]["cellBorderRandom"])
        parameters["segmentation"]["seeding"]["from"]["cellContentRandom"] = int(parameters["segmentation"]["seeding"]["from"]["cellContentRandom"])
        parameters["segmentation"]["background"]["blurSteps"] = int(parameters["segmentation"]["background"]["blurSteps"])
        parameters["segmentation"]["steps"] = int(parameters["segmentation"]["steps"])
        parameters["segmentation"]["stars"]["points"] = int(parameters["segmentation"]["stars"]["points"])
        return parameters

    sfrom = lambda x: max(0, segmentation_precision - x)
    segmentation_precision = min(20, segmentation_precision)
    if segmentation_precision <= 0:
        parameters["segmentation"]["steps"] = 0
    elif segmentation_precision <= 6:
        parameters["segmentation"]["steps"] = 1
    else:
        parameters["segmentation"]["steps"] = min(10, segmentation_precision - 5)

    parameters["segmentation"]["stars"]["points"] = 8 + max(segmentation_precision - 2, 0) * 4

    parameters["segmentation"]["maxFreeBorder"] = \
        max(0.4, 0.7 * 16 / max(16, parameters["segmentation"]["stars"]["points"]))

    parameters["segmentation"]["seeding"]["from"]["houghTransform"] = int(segmentation_precision == 1)
    parameters["segmentation"]["seeding"]["from"]["cellBorder"] = int(segmentation_precision >= 2)
    parameters["segmentation"]["seeding"]["from"]["cellBorderRandom"] = sfrom(14)
    parameters["segmentation"]["seeding"]["from"]["cellContent"] = int(segmentation_precision >= 11)
    parameters["segmentation"]["seeding"]["from"]["cellContentRandom"] = min(4, sfrom(12))
    parameters["segmentation"]["seeding"]["from"]["cellBorderRemovingCurrSegments"] = \
        int(segmentation_precision >= 11)
    parameters["segmentation"]["seeding"]["from"]["cellBorderRemovingCurrSegmentsRandom"] = min(4, sfrom(16))
    parameters["segmentation"]["seeding"]["from"]["cellContentRemovingCurrSegments"] = \
        int(segmentation_precision >= 7)
    parameters["segmentation"]["seeding"]["from"]["cellContentRemovingCurrSegmentsRandom"] = min(4, sfrom(12))
    parameters["segmentation"]["seeding"]["from"]["snakesCentroids"] = int(segmentation_precision >= 9)
    parameters["segmentation"]["seeding"]["from"]["snakesCentroidsRandom"] = min(4, sfrom(14))

    parameters["segmentation"]["stars"]["step"] = 0.0067 * max(1, (1 + (15 - segmentation_precision) / 2))

    if segmentation_precision <= 9:
        size_weight_multiplier = np.array([1])
    elif segmentation_precision <= 11:
        size_weight_multiplier = np.array([0.8, 1.25])
    elif segmentation_precision <= 13:
        size_weight_multiplier = np.array([0.6, 1, 1.6])
    elif segmentation_precision <= 15:
        size_weight_multiplier = np.array([0.5, 0.8, 1.3, 2])
    elif segmentation_precision <= 17:
        size_weight_multiplier = np.array([0.35, 0.5, 0.8, 1.3, 2, 3])
    else:
        size_weight_multiplier = np.array([0.25, 0.35, 0.5, 0.8, 1.3, 2, 3, 5, 8])

    parameters["segmentation"]["stars"]["sizeWeight"] = \
        np.average(parameters["segmentation"]["stars"]["sizeWeight"]) * size_weight_multiplier / \
        np.average(size_weight_multiplier)
    parameters["segmentation"]["stars"]["sizeWeight"] = list(parameters["segmentation"]["stars"]["sizeWeight"])

    parameters["segmentation"]["foreground"]["pickyDetection"] = segmentation_precision > 8
    if "tracking" not in parameters:
        parameters["tracking"] = {}
    parameters["tracking"]["iterations"] = max(1, segmentation_precision * 5 - 25)

    return parameters