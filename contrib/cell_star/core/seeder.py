# -*- coding: utf-8 -*-
"""
Seeder is responsible for finding places where might be cells.
Date: 2013-2016
Website: http://cellstar-algorithm.org/
"""
import random

from contrib.cell_star.core.seed import Seed
from contrib.cell_star.utils.calc_util import *
from contrib.cell_star.utils.image_util import *


class Seeder(object):
    """
    Find places for seeds based on images and previously found snakes.
    """

    def __init__(self, images, parameters):
        """
        @type images: core.image_repo.ImageRepo
        @type parameters: dict
        """
        self.images = images
        np.random.seed(abs(hash(abs(np.sum(images.image)))))
        random.seed(abs(np.sum(images.image)))
        self.parameters = parameters
        self.cluster_min_distance = self.parameters["segmentation"]["seeding"]["minDistance"] \
                                    * self.parameters["segmentation"]["avgCellDiameter"]
        self.random_radius = self.parameters["segmentation"]["seeding"]["randomDiskRadius"] \
                             * self.parameters["segmentation"]["avgCellDiameter"]

    def cluster_seeds(self, seeds):
        """
        @type seeds: list[Seed]
        """
        origin = 'cluster'
        if len(seeds) > 0:
            origin = seeds[0].origin
        seeds = np.array(map(lambda s: (s.x, s.y), seeds))
        my_inf = \
            np.inf if seeds.dtype.kind == 'f' else np.iinfo(seeds.dtype).max
        while len(seeds) > 1:
            #
            # find p1 and p2 closest to each other - p1 is closest to p2
            # and p2 is closest to p1
            #
            p1 = np.arange(seeds.shape[0])
            d = seeds[:, np.newaxis, :] - seeds[np.newaxis, :, :]
            d2 = np.sum(d * d, 2)
            d2[p1, p1] = my_inf
            p2 = np.argmin(d2, 0)
            #
            # Eliminate p1 / p2 if p1 is closest to p2 and p2 is closest to
            # someone else
            #
            good = p1 == p2[p2]
            p1, p2 = p1[good], p2[good]
            #
            # Eliminate p1 / p2 if p1 > p2 (get rid of (2, 1) since there is
            # a (1, 2)
            #
            good = p1 < p2
            p1, p2 = p1[good], p2[good]
            #
            # Eliminate p1 / p2 if < self.cluster_min_distance
            #
            good = d2[p1, p2] < self.cluster_min_distance
            p1, p2 = p1[good], p2[good]
            if len(p1) == 0:
                break
            #
            # coalesce
            #
            new_seeds = (seeds[p1, :] + seeds[p2, :]) / 2.0
            to_keep = np.ones(seeds.shape[0], bool)
            to_keep[p1] = False
            to_keep[p2] = False
            seeds = np.vstack((seeds[to_keep, :], new_seeds))

        seed_points = point_list_as_seeds(seeds, origin)
        return seed_points

    @staticmethod
    def rand_seeds(max_random_radius, times, seeds):
        """
        @type seeds: list[Seed]
        """
        seeds_number = len(seeds)
        px = [s.x for s in seeds] * times
        py = [s.y for s in seeds] * times

        random_angle = np.random.random(seeds_number * times) * 2 * math.pi
        random_radius = np.random.random(seeds_number * times) * max_random_radius

        rpx = px + random_radius * np.cos(random_angle)
        rpy = py + random_radius * np.sin(random_angle)

        return [Seed(x, y, 'rand') for x, y in zip(rpx, rpy)]

    def find_seeds_from_border_or_content(self, image, foreground_mask, segments, mode):
        """
        Finds seeds from given image.
        @param image: image (border or content) from which seeds are being extracted
        @type image: np.ndarray
        @param foreground_mask: binary foreground mask
        @type foreground_mask: np.ndarray
        @param segment: array with segments marked
        @type segments: np.ndarray
        @param mode: 'border' or 'content' determining if look for maxima or minima
        """

        im_name = ''
        origin = ''

        if mode == 'border':
            blur = self.parameters["segmentation"]["seeding"]["BorderBlur"] \
                   * self.parameters["segmentation"]["avgCellDiameter"]
            im_name = 'border' + im_name
            image = set_image_border(image, 1)
            excl_value = 1
        else:
            blur = self.parameters["segmentation"]["seeding"]["ContentBlur"] \
                   * self.parameters["segmentation"]["avgCellDiameter"]
            im_name = 'content' + im_name
            excl_value = 0

        excluding = segments is not None
        if excluding:
            image = exclude_segments(image, segments, excl_value)
            origin = ' no segments'
            im_name += ' excluding current segments'

        origin = im_name + origin

        blurred = self.images.get_blurred(image, blur, cache_result=not excluding)

        if mode == 'border':
            blurred = 1 - blurred

        maxima = find_maxima(blurred) * foreground_mask

        maxima_coords = sp.nonzero(maxima)
        seeds = zip(maxima_coords[1], maxima_coords[0])

        seed_points = point_list_as_seeds(seeds, origin)
        seed_points = self.cluster_seeds(seed_points)

        return seed_points

    def find_seeds_from_snakes(self, snakes):
        """
        Finds seeds from snakes centroids
        @param snakes: Grown snakes from previous frame
        @type snakes: list[Snake]
        """
        return point_list_as_seeds([snake.centroid for snake in snakes], 'snake centroid')

    def find_seeds(self, snakes, all_seeds, exclude_current_segments):
        seeds = []

        # First iteration of segmentation.
        if not exclude_current_segments:
            if self.parameters["segmentation"]["seeding"]["from"]["cellBorder"]:
                new_seeds = self.find_seeds_from_border_or_content(self.images.brighter, self.images.foreground_mask,
                                                                   None, 'border')
                new_seeds += self.rand_seeds(
                    self.random_radius,
                    self.parameters["segmentation"]["seeding"]["from"]["cellBorderRandom"],
                    new_seeds
                )

            seeds += new_seeds

            if self.parameters["segmentation"]["seeding"]["from"]["cellContent"]:
                new_seeds = self.find_seeds_from_border_or_content(self.images.darker, self.images.foreground_mask,
                                                                   None, 'content')
                new_seeds += self.rand_seeds(
                    self.random_radius,
                    self.parameters["segmentation"]["seeding"]["from"]["cellContentRandom"],
                    new_seeds
                )

                seeds += new_seeds

        elif len(snakes) > 0:
            if self.parameters["segmentation"]["seeding"]["from"]["cellBorderRemovingCurrSegments"]:
                new_seeds = self.find_seeds_from_border_or_content(self.images.brighter, self.images.foreground_mask,
                                                                   self.images.segmentation, 'border')
                new_seeds += self.rand_seeds(
                    self.random_radius,
                    self.parameters["segmentation"]["seeding"]["from"]["cellBorderRemovingCurrSegmentsRandom"],
                    new_seeds
                )

                seeds += new_seeds

            if self.parameters["segmentation"]["seeding"]["from"]["cellContentRemovingCurrSegments"]:
                new_seeds = self.find_seeds_from_border_or_content(self.images.darker, self.images.foreground_mask,
                                                                   self.images.segmentation, 'content')
                new_seeds += self.rand_seeds(
                    self.random_radius,
                    self.parameters["segmentation"]["seeding"]["from"]["cellContentRemovingCurrSegmentsRandom"],
                    new_seeds
                )

                seeds += new_seeds

            # Seeds from existing contours
            if self.parameters["segmentation"]["seeding"]["from"]["snakesCentroids"]:
                new_seeds = self.find_seeds_from_snakes(snakes)
                new_seeds += self.rand_seeds(
                    self.random_radius,
                    self.parameters["segmentation"]["seeding"]["from"]["snakesCentroidsRandom"],
                    new_seeds
                )
                seeds += new_seeds

        # TODO: filter seeds using foreground mask ?
        # seeds = [seed for seed in seeds if self.images.foreground_mask[seed.y, seed.x]]

        seeds = self._filter_seeds(seeds, all_seeds)

        return seeds

    def _filter_seeds(self, seeds, all_seeds):
        """
        @param seeds:
        @type seeds: [contrib.cell_star.core.seed.Seed]
        @return:
        """
        distance = self.parameters["segmentation"]["stars"]["step"] * self.parameters["segmentation"]["avgCellDiameter"]
        distance = float(max(distance, 0.5))  # not less than half of pixel length
        ok_seeds = np.array([False for seed in seeds])

        # TODO: obecnie parametr ustawiony na -1 - wykryć gdzie jest konfigurowany w MATLABIE
        # Wygląda na to, że jest to lista wymiarów klatki/zdjęcia
        # if self.parameters["segmentation"]["transform"]["originalImDim"] > 0:
        grid_size = int(round(max(self.parameters["segmentation"]["transform"]["originalImDim"]) * 1.1 / distance))
        im_x = self.parameters["segmentation"]["transform"]["originalImDim"][1]
        im_y = self.parameters["segmentation"]["transform"]["originalImDim"][0]
        # else:
        #    grid_size = 10

        # Create grid
        seeds_grid = SeedGrid(grid_size)

        # Fill grid with previous seeds
        for seed in all_seeds:
            x = int(round(seed.x / distance))
            y = int(round(seed.y / distance))
            for xx in [x - 1, x, x + 1]:
                for yy in [y - 1, y, y + 1]:
                    seeds_grid.add_seed(xx, yy, seed)

        # Fill grid with current seeds
        for i in xrange(len(seeds)):
            seed = seeds[i]
            x = max(0, int(round(seed.x / distance)))
            y = max(0, int(round(seed.y / distance)))

            # Validate seed if it lies inside grid
            if seeds_grid.inside(x, y):
                ok_seeds[i] = seed_is_new(seed, seeds_grid.get(x, y), distance)

            for xx in [x - 1, x, x + 1]:
                for yy in [y - 1, y, y + 1]:
                    seeds_grid.add_seed(xx, yy, seed)

        # Remove seeds in image borders
        seeds_x_ok = np.array([im_x - 0.5 > seed.x > 0.5 for seed in seeds])
        seeds_y_ok = np.array([im_y - 0.5 > seed.y > 0.5 for seed in seeds])

        # Filtered seeds boolean vector
        ok_seeds = np.logical_and(ok_seeds, np.logical_and(seeds_y_ok, seeds_x_ok))

        return [seeds[i] for i in range(len(seeds)) if ok_seeds[i]]


def seed_is_new(_seed, all_seeds, distance):
    for seed in all_seeds:
        if euclidean_norm(_seed.as_xy(), seed.as_xy()) < distance:
            return False

    return True


def point_list_as_seeds(points, origin):
    return [Seed(point[0], point[1], origin) for point in points]


class SeedGrid(object):
    def __init__(self, size=10):
        self._size_x = size
        self._size_y = size
        self._grid = []
        self._init_grid()

    def _init_grid(self):
        self._grid = np.empty((self._size_x, self._size_y), dtype=np.object)
        self._grid.fill([])

    def _resize(self, to_x, to_y):
        self._size_y = max(self._size_y, to_y)
        self._size_x = max(self._size_x, to_x)

        self._grid = np.resize(self._grid, (self._size_x, self._size_y))

    def get(self, x, y):
        return self._grid[x][y]

    def size(self):
        return self._size_x, self._size_y

    def add_seed(self, x, y, seed):
        if x < 0 or y < 0:
            return

        if x >= self._size_x or y >= self._size_y:
            self._resize(x + 1, y + 1)

        self._grid[x][y] = self._grid[x][y] + [seed]

    def inside(self, x, y):
        return x < self._size_x and y < self._size_y
