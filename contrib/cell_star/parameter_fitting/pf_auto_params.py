# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

import numpy as np

parameters_range = {"brightnessWeight": (-0.4, 0.4),
                    "cumBrightnessWeight": (0, 500),
                    "gradientWeight": (-30, 30),
                    "sizeWeight": (10, 300),
                    "smoothness": (2, 15),
                    "borderThickness": (0.0, 1.0)
}

rank_parameters_range = {"avgBorderBrightnessWeight": (0, 600),
                         "avgInnerBrightnessWeight": (-30, 30),
                         "avgInnerDarknessWeight": (-20, 20),
                         "logAreaBonus": (4, 30),
                         "maxInnerBrightnessWeight": (-10, 30),
                         # "maxRank": (5, 300),
                         # "stickingWeight": (0, 6)  # cannot calculate entropy for mutants
}

#
#
# PARAMETERS ENCODE DECODE
#
#

def pf_parameters_encode(parameters):
    """
    brightnessWeight: 0.0442 - jak bardzo lubimy jasnosc na krancach komorki
    cumBrightnessWeight: 304.45 - jak bardzo nie lubimy jasnego w �rodku komorki
    gradientWeight: 15.482 - jak bardzo lubimy gradient na kra�cach komorki
    sizeWeight: 189.4082 ! może być lista, wtedy mean - jak bardzo lubimy du�e kom�rki
    smoothness: 7.0 - jaki jest max

    @param parameters: dictionary segmentation.stars
    """
    parameters = parameters["segmentation"]["stars"]
    point = []
    for name, (vmin, vmax) in parameters_range.iteritems():
        val = parameters[name]
        if name == "sizeWeight":
            if not isinstance(val, float):
                val = np.mean(val)
        # trim_val = max(vmin, min(vmax, val))
        # point.append((trim_val - vmin) / float(vmax - vmin))
        point.append(val)
    # should be scaled to go from 0-1
    return point


def pf_parameters_decode(param_vector, org_size_weights_list):
    """
    sizeWeight is one number (mean of the future list)
    @type param_vector: numpy.ndarray
    @return:
    """
    parameters = {}
    for (name, (vmin, vmax)), val in zip(parameters_range.iteritems(), param_vector):
        # val = min(1, max(0, val))
        # rescaled = vmin + val * (vmax - vmin)
        rescaled = val
        if name == "sizeWeight":
            rescaled = list(np.array(org_size_weights_list) * (rescaled/np.mean(org_size_weights_list)))
        elif name == "borderThickness":
            rescaled = max(0.001, val)
        parameters[name] = rescaled
    return parameters


def pf_rank_parameters_encode(parameters):
    """
    # Set: config.yaml
    # Usage: snake.py - 2 times
    avgBorderBrightnessWeight: 300 # OPT
    # Set: config.yaml
    # Usage: snake.py - 2 times
    avgInnerBrightnessWeight: 10 # OPT
    # Set: config.yaml
    # Usage: snake.py - 2 times - as multiplier - zeroes avg_inner_darkness in calculation of rank
    avgInnerDarknessWeight: 0 # OPT
    # Set: config.yaml
    # Usage: snake.py - 2 times
    logAreaBonus: 18 # OPT
    # Set: config.yaml
    # Usage: snake.py - 2 times
    maxInnerBrightnessWeight: 10 # OPT
    # Set: config.yaml
    # Usage: snake_filter.py - 1 time - actually 0 is meaningfull (!)
    maxRank: 100 # OPT
    # Set: config.yaml
    # Usage: snake.py - 1 time - as ranking weight
    stickingWeight: 60 # OPT
    @param parameters: dictionary all params
    """
    parameters = parameters["segmentation"]["ranking"]
    point = []
    for name, (vmin, vmax) in rank_parameters_range.iteritems():
        val = parameters[name]
        trim_val = max(vmin, min(vmax, val))
        if vmax - vmin == 0:
            point.append(0)
        else:
            point.append((trim_val - vmin) / float(vmax - vmin))
    return point


def pf_rank_parameters_decode(param_vector):
    """
    @type param_vector: numpy.ndarray
    @return: only ranking parameters as a dict
    """
    parameters = {}
    for (name, (vmin, vmax)), val in zip(rank_parameters_range.iteritems(), param_vector):
        val = min(1, max(0, val))
        rescaled = vmin + val * (vmax - vmin)
        parameters[name] = rescaled
    parameters["stickingWeight"] = 0
    return parameters
