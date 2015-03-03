# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

# External imports
import logging
import numpy as np
import math
# Internal imports
from contrib.cell_star.core.snake import Snake
from contrib.cell_star.utils.image_util import image_show, draw_snakes


class SnakeFilter(object):

    def __init__(self, parameters, images):
        """
        @type parameters: dict
        @param parameters:
        @type images: core.image_repo.ImageRepo
        @param images:
        """
        self.parameters = parameters
        self.images = images

    def filter(self, snakes):
        """
        @type snakes: list of Snake
        @param snakes:
        @rtype: list of Snake
        """
        logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.DEBUG)
        logger = logging.getLogger(__package__)
        log_message = "Discarding snake {0} for {1}: {2}"

        original = self.images.image
        new_snakes = []
        segments = np.zeros(original.shape, dtype=int)

        if len(snakes) > 0:
            snakerank = [snake.rank for snake in snakes]
            snakes_indices = [i[0] for i in sorted(enumerate(snakerank), key=lambda x:x[1])]
            snakes = sorted(snakes, key=lambda ss: ss.rank)

            draw_snakes(original * 0.2, snakes)

            current_accepted_snake_index = 1
            for i in xrange(len(snakes)):
                curr_snake = snakes[i]

                if False:  # curr_snake.rank > self.parameters["segmentation"]["ranking"]["maxRank"]:
                    #  filtering by rank can be done by user in another module we only export it as a measurement
                    logger.debug(log_message.format(snakes_indices[i], 'too high rank', curr_snake.rank))
                else:
                    xy = curr_snake.in_polygon_xy
                    in_polygon = curr_snake.in_polygon
                    xy = np.array([xy, np.array(xy) + in_polygon.shape]).flatten()

                    area = np.count_nonzero(in_polygon)
                    if self.parameters["segmentation"]["minSize"] > area:
                        logger.debug(log_message.format(snakes_indices[i], 'too small', area))
                        continue
                    if self.parameters["segmentation"]["maxSize"] < area:
                        logger.debug(log_message.format(snakes_indices[i], 'too big', area))
                        continue

                    dilated_segments = segments[xy[0]:xy[2], xy[1]:xy[3]]
                    overlap_area = np.count_nonzero(dilated_segments[in_polygon])
                    overlap = float(overlap_area) / curr_snake.area

                    if overlap > self.parameters["segmentation"]["maxOverlap"]:
                        logger.debug(log_message.format(snakes_indices[i], 'too much overlapping', overlap))
                    else:
                        in2 = np.logical_and(in_polygon, dilated_segments == 0)
                        tmp = in2[self.images.cell_content_mask[xy[0]:xy[2], xy[1]:xy[3]]]
                        curr_snake.area = in2.sum() + Snake.epsilon
                        avg_inner_darkness = float(np.count_nonzero(tmp)) / float(curr_snake.area)
                        if avg_inner_darkness < self.parameters["segmentation"]["minAvgInnerDarkness"]:
                            logger.debug(log_message.format(snakes_indices[i], 'too low inner darkness', '...'))
                        else:
                            if curr_snake.area > (self.parameters["segmentation"]["maxArea"] * self.parameters["segmentation"]["avgCellDiameter"]**2 * math.pi / 4):
                                logger.debug(log_message.format(snakes_indices[i], 'too big area', str(curr_snake.area)))
                            else:
                                if curr_snake.area < (self.parameters["segmentation"]["minArea"] * self.parameters["segmentation"]["avgCellDiameter"]**2 * math.pi / 4):
                                    logger.debug(log_message.format(snakes_indices[i], 'too small area:', str(curr_snake.area)))
                                else:
                                    max_free_border = self.parameters["segmentation"]["stars"]["points"] * self.parameters["segmentation"]["maxFreeBorder"]
                                    if curr_snake.max_contiguous_free_border > max_free_border:
                                        logger.debug(log_message.format(snakes_indices[i],
                                                                        'too long contiguous free border',
                                                                        str(curr_snake.max_contiguous_free_border) +
                                                                        'over' + str(max_free_border)))
                                    else:
                                        dilated_segments[[in2]] = current_accepted_snake_index
                                        segments[xy[0]:xy[2], xy[1]:xy[3]] = dilated_segments
                                        new_snakes.append(curr_snake)
                                        current_accepted_snake_index += 1

        self.images._segmentation = segments
        return new_snakes