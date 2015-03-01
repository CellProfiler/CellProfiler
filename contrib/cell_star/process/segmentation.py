# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

# External imports
from copy import copy
import sys
# Internal imports
from contrib.cell_star.utils.params_util import *
from contrib.cell_star.core.image_repo import ImageRepo
from contrib.cell_star.utils.params_util import default_parameters
from contrib.cell_star.utils import image_util
from contrib.cell_star.core.seeder import Seeder
from contrib.cell_star.core.seed import Seed
from contrib.cell_star.core.snake import Snake
from contrib.cell_star.core.snake_filter import SnakeFilter
from contrib.cell_star.core.polar_transform import PolarTransform
from contrib.cell_star.parameter_fitting.pf_auto_params import rank_parameters_range as rank_auto_params
from contrib.cell_star.parameter_fitting.pf_auto_params import parameters_range as snake_auto_params
from contrib.cell_star.utils.python_util import memory_profile, speed_profile


class Segmentation(object):
    def __init__(self, segmentation_precision=7, avg_cell_diameter=35, debug_level=0):
        self.parameters = default_parameters(segmentation_precision, avg_cell_diameter)
        self.images = None
        self.all_seeds = []
        self.seeds = []
        # seeds from which we already have snakes
        # PL: Seedy, z których już wyrosły snake'i
        self.grown_seeds = set()
        self.snakes = []
        self.new_snakes = []
        self._seeder = None
        self._filter = None
        self.polar_transform = PolarTransform.instance(self.parameters["segmentation"]["avgCellDiameter"],
                                                       self.parameters["segmentation"]["stars"]["points"],
                                                       self.parameters["segmentation"]["stars"]["step"],
                                                       self.parameters["segmentation"]["stars"]["maxSize"])
        self.debug_output_image_path = None

    def clear_lists(self):
        self.all_seeds = []
        self.seeds = []
        self.grown_seeds = set()
        self.snakes = []
        self.new_snakes = []

    @property
    def seeder(self):
        if self._seeder is None:
            self.init_seeder()

        return self._seeder

    @property
    def filter(self):
        if self._filter is None:
            self.init_filter()

        return self._filter

    def set_frame(self, frame):
        # Extract previous background
        prev_background = None
        if self.images is not None:
            prev_background = self.images.background
        # Initialize new image repository for new frame
        self.images = ImageRepo(frame, self.parameters)
        # One background per whole segmentation
        if prev_background is not None:
            self.set_background(prev_background)
        # Update image dimensions parameter
        try:
            self.parameters["segmentation"]["transform"]["originalImDim"] = frame.shape
        except KeyError:
            self.parameters["segmentation"]["transform"] = {}
            self.parameters["segmentation"]["transform"]["originalImDim"] = frame.shape

    def set_background(self, background):
        self.images._background = background

    def init_seeder(self):
        self._seeder = Seeder(self.images, self.parameters)

    def init_filter(self):
        self._filter = SnakeFilter(self.parameters, self.images)

    def decode_auto_params(self, text):
        """
        Decode automatic parameters from text and apply to self.

        @param text: parameters denoted as python list
        @:return true if parsing was successful
        """
        new_stars = copy(self.parameters["segmentation"]["stars"])
        new_ranking = copy(self.parameters["segmentation"]["ranking"])
        try:
            exec "all_params=" + text
            snake_params = all_params[0]
            rank_params = all_params[1]
            if len(snake_params) != len(snake_auto_params) or len(rank_params) != len(rank_auto_params):
                raise Exception("text invalid: list size not compatible")


            for name in sorted(snake_auto_params.keys()):
                val = snake_params[0]
                if name == "sizeWeight":  # value to list
                    original = self.parameters["segmentation"]["stars"]["sizeWeight"]
                    val = list(np.array(original) * (val/np.mean(original)))

                new_stars[name] = val
                snake_params = snake_params[1:]

            for name in sorted(rank_auto_params.keys()):
                new_ranking[name] = rank_params[0]
                rank_params = rank_params[1:]
        except:
            return False

        self.parameters["segmentation"]["stars"] = new_stars
        self.parameters["segmentation"]["ranking"] = new_ranking
        return True

    @staticmethod
    def encode_auto_params_from_all_params(parameters):
        snake_auto_params_values = []
        for name in sorted(snake_auto_params.keys()):
            val = parameters["segmentation"]["stars"][name]
            if name == "sizeWeight":  # list to mean value
                original = parameters["segmentation"]["stars"]["sizeWeight"]
                val = np.mean(original)
            snake_auto_params_values.append(val)

        rank_auto_params_values = [parameters["segmentation"]["ranking"][name]
                                   for name in sorted(rank_auto_params.keys())]
        auto_values_list = [snake_auto_params_values, rank_auto_params_values]

        return str(auto_values_list)

    def encode_auto_params(self):
        return Segmentation.encode_auto_params_from_all_params(self.parameters)

    def pre_process(self):
        # Condition always false because property 'background' is never None, but important :)
        if self.images.background is None:
            self.images.calculate_background()

        self.images.calculate_brighter_original()
        self.images.calculate_darker_original()
        self.images.calculate_clean_original()
        self.images.calculate_forebackground_masks()

        self.images.calculate_clean()
        self.images.calculate_brighter()
        self.images.calculate_darker()

        self.images.calculate_cell_border_content_mask()

    def find_seeds(self, exclude):
        self.seeds = self.seeder.find_seeds(self.snakes, self.all_seeds, exclude_current_segments=exclude)
        self.all_seeds += self.seeds

    def snakes_from_seeds(self):
        self.new_snakes = [
            Snake.create_from_seed(
                self.parameters, seed, self.parameters["segmentation"]["stars"]["points"], self.images
            )
            for seed in self.seeds if seed not in self.grown_seeds
        ]
        for seed in self.seeds:
            if seed not in self.grown_seeds:
                self.grown_seeds.add(seed)

    def grow_snakes(self):
        new_snakes = []
        size_weights = self.parameters["segmentation"]["stars"]["sizeWeight"]
        logger.debug("%d snakes seeds to grow with %d weights options -> %d snakes to calculate"%(len(self.new_snakes), len(size_weights), len(self.new_snakes) * len(size_weights)))
        for snake in self.new_snakes:
            best_snake = None
            for weight in size_weights:
                curr_snake = copy(snake)

                curr_snake.star_grow(weight, self.polar_transform)
                curr_snake.calculate_properties_vec(self.polar_transform)

                if best_snake is None:
                    best_snake = curr_snake
                else:
                    if curr_snake.rank < best_snake.rank:
                        best_snake = curr_snake

            new_snakes.append(best_snake)

        self.new_snakes = new_snakes

    def evaluate_snakes(self):
        for snake in self.new_snakes:
            snake.calculate_properties_vec(self.polar_transform)

    def filter_snakes(self):
        self.snakes = self.filter.filter(self.snakes + self.new_snakes)
        self.new_snakes = []

    def debug_images(self):
        if self.debug_output_image_path is not None:
            image_util.debug_image_path = self.debug_output_image_path
        image_util.image_save(self.images.background, "background")
        image_util.image_save(self.images.brighter, "brighter")
        image_util.image_save(self.images.brighter_original, "brighter_original")
        image_util.image_save(self.images.darker, "darker")
        image_util.image_save(self.images.darker_original, "darker_original")
        image_util.image_save(self.images.cell_content_mask, "cell_content_mask")
        image_util.image_save(self.images.cell_border_mask, "cell_border_mask")
        image_util.image_save(self.images.foreground_mask, "foreground_mask")
        image_util.image_save(self.images.image_back_difference, "image_back_difference")
        pass

    def debug_seeds(self, step):
        if self.debug_output_image_path is not None:
            image_util.debug_image_path = self.debug_output_image_path
        image_util.draw_seeds(self.all_seeds, self.images.image, title=str(step))

    def run_one_step(self, step):
        logger.debug("find_seeds")
        self.find_seeds(step > 0)
        self.debug_seeds(step)
        logger.debug("snake_from_seeds")
        self.snakes_from_seeds()
        logger.debug("grow_snakes")
        self.grow_snakes()
        logger.debug("filter_snakes")
        image_util.draw_snakes(self.images.image, self.snakes + self.new_snakes, it=step)
        self.filter_snakes()
        logger.debug("done")

    #@memory_profile
    def run_segmentation(self):
        logger.debug("preproces...")
        self.pre_process()
        self.debug_images()
        for step in range(self.parameters["segmentation"]["steps"]):
            self.run_one_step(step)
        image_util.image_show(self.images.image, 1)
        # image_util.image_show(self.images.image + (self.images.segmentation > 0), 1)
        return self.images.segmentation, self.snakes

    def formatted_result(self, result_format):
        return result_format(self.images.segmentation, self.snakes)