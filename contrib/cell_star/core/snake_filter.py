# -*- coding: utf-8 -*-
"""
SnakeFilter is responsible for ranking and filtering out contours that as incorrect or overlap better ones.
Date: 2013-2016
Website: http://cellstar-algorithm.org/
"""

import logging
import math

import numpy as np

from contrib.cell_star.core.snake import Snake


class SnakeFilter(object):
    """
    Order snakes based on their ranking and checks them for violating other constraints.
    Discard snakes that overlap with already approved ones.
    """

    def __init__(self, parameters, images):
        """
        @type parameters: dict
        @type images: core.image_repo.ImageRepo
        """
        self.parameters = parameters
        self.images = images

    def filter(self, snakes):
        """
        @type snakes: list[Snake]
        @rtype: list[Snake]
        """
        logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.DEBUG)
        logger = logging.getLogger(__package__)
        log_message = "Discarding snake {0} for {1}: {2}"

        original = self.images.image
        filtered_snakes = []
        segments = np.zeros(original.shape, dtype=int)

        if len(snakes) > 0:
            snakes_sorted = sorted(enumerate(snakes), key=lambda x: x[1].rank)
            current_accepted_snake_index = 1
            for i in xrange(len(snakes_sorted)):
                curr_snake = snakes_sorted[i][1]
                snake_index = snakes_sorted[i][0]

                local_snake = curr_snake.in_polygon
                sxy = curr_snake.in_polygon_slice
                local_segments = segments[sxy]

                overlap_area = np.count_nonzero(np.logical_and(local_segments, local_snake))
                overlap = float(overlap_area) / curr_snake.area

                if overlap > self.parameters["segmentation"]["maxOverlap"]:
                    logger.debug(log_message.format(snake_index, 'too much overlapping', overlap))
                else:
                    vacant_snake = np.logical_and(local_snake, local_segments == 0)
                    vacant_cell_content = vacant_snake[self.images.cell_content_mask[sxy]]
                    curr_snake.area = np.count_nonzero(vacant_snake) + Snake.epsilon
                    avg_inner_darkness = float(np.count_nonzero(vacant_cell_content)) / float(curr_snake.area)
                    if avg_inner_darkness < self.parameters["segmentation"]["minAvgInnerDarkness"]:
                        logger.debug(log_message.format(snake_index, 'too low inner darkness', '...'))
                    else:
                        if curr_snake.area > (self.parameters["segmentation"]["maxArea"] * self.parameters["segmentation"]["avgCellDiameter"]**2 * math.pi / 4):
                            logger.debug(log_message.format(snake_index, 'too big area', str(curr_snake.area)))
                        else:
                            if curr_snake.area < (self.parameters["segmentation"]["minArea"] * self.parameters["segmentation"]["avgCellDiameter"]**2 * math.pi / 4):
                                logger.debug(log_message.format(snake_index, 'too small area:', str(curr_snake.area)))
                            else:
                                max_free_border = self.parameters["segmentation"]["stars"]["points"] * self.parameters["segmentation"]["maxFreeBorder"]
                                if curr_snake.max_contiguous_free_border > max_free_border:
                                    logger.debug(log_message.format(snake_index,
                                                                    'too long contiguous free border',
                                                                    str(curr_snake.max_contiguous_free_border) +
                                                                    'over' + str(max_free_border)))
                                else:
                                    local_segments[[vacant_snake]] = current_accepted_snake_index
                                    filtered_snakes.append(curr_snake)
                                    current_accepted_snake_index += 1

        self.images._segmentation = segments
        return filtered_snakes
