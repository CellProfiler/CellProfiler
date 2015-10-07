# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

# External imports
import random
from copy import copy
# Internal imports
# Objects
from contrib.cell_star.core.seed import Seed
# Utils
from contrib.cell_star.utils.cluster_util import *
from contrib.cell_star.utils.calc_util import *
from contrib.cell_star.utils.image_util import *


class Seeder(object):

    def __init__(self, images, parameters):
        self.images = images
        random.seed(abs(np.sum(images.image)))
        self.parameters = parameters
        self.cluster_min_distance = self.parameters["segmentation"]["seeding"]["minDistance"] \
            * self.parameters["segmentation"]["avgCellDiameter"]
        self.random_radius = self.parameters["segmentation"]["seeding"]["randomDiskRadius"] \
            * self.parameters["segmentation"]["avgCellDiameter"]

    def cluster_seeds(self, seeds):
        origin = 'cluster'
        if len(seeds) > 0:
            origin = seeds[0].origin
        seeds = map(lambda s: (s.x, s.y), seeds)
        # Distance matrix
        distance_matrix = square_form(pdist(seeds, 'euclidean'))
        # print "Seeds to cluster: " + str(len(seeds))
        while np.count_nonzero((distance_matrix > 0) & (distance_matrix < self.cluster_min_distance)) > 0:
            # Find nonzero values in distance_matrix
            ind = np.nonzero(distance_matrix)
            # Find minimum of nonzero values in distance_matrix
            minimal_distance = min(distance_matrix[ind])
            # Find pair of arrays with indices of elements from distance_matrix equal to minimum
            k = np.nonzero(distance_matrix * (distance_matrix == minimal_distance))
            # First index of first array -> indicating seed 1
            i = k[0][0]
            seed1 = seeds[i]
            # First index of second array -> indicating seed 2
            j = k[1][0]
            seed2 = seeds[j]
            # Create new seed as arithmetic mean of seed 1 and seed 2
            # (two first seeds found with minimal distance below threshold)
            new_seed = (float(seed1[0] + seed2[0]) / 2.0, float(seed1[1] + seed2[1]) / 2.0)
            # Replace seed 1 with new seed
            seeds[i] = new_seed
            # Remove seed 2 """
            seeds = seeds[:j] + seeds[j + 1:]
            distance_matrix = np.delete(distance_matrix, j, 0)
            distance_matrix = np.delete(distance_matrix, j, 1)
            # Recalculate distance matrix
            for k in xrange(0, len(seeds)):
                if k != i:
                    dist = euclidean_norm(seeds[i], seeds[k])
                    distance_matrix[i][k] = dist
                    distance_matrix[k][i] = dist
                    # everything connected to seeds[j] is to die
                    # distance_matrix = squareform(pdist(seeds, 'euclidean'))

        seed_points = point_list_as_seeds(seeds, origin)
        return seed_points

    @staticmethod
    def rand_seeds(random_radius, times, seeds):
            # Wprowadzam zmianę: kopia seedów tak jest w oryginale. Obecnie zmieniana jest także lista seeds (python)
            rand_seeds = []
            for j in xrange(times):
                new_seeds = copy(seeds)
                angles = []
                radius = []
                for _ in xrange(len(new_seeds)):
                    angles.append(random.random() * 2 * math.pi)
                    radius.append(random.random() * random_radius)
                for i in xrange(len(seeds)):
                    x = new_seeds[i].x + radius[i] * math.cos(angles[i])
                    y = new_seeds[i].y + radius[i] * math.sin(angles[i])
                    f = new_seeds[i].origin + '_rand'
                    new_seeds[i] = Seed(x, y, f)
                rand_seeds = rand_seeds + new_seeds

            return rand_seeds

    def find_seeds_from_border_or_content(self, image, foreground_mask, segments, mode, excl=False):
            """
            Finds seeds from given image
            @param image: image (border or content) from which seeds are being extracted
            @param foreground_mask: binary foreground mask
            @param mode: 'border' or 'content' determining if look for maxima or minima
            @param excl: former 'RemovingCurrSegments'
            """

            #seeds = []
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

            if excl:
                image = exclude_segments(image, segments, excl_value)
                origin = ' no segments'
                im_name += ' excluding current segments'

            origin = im_name + origin

            blurred = self.images.get_blurred(image, blur)

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
            """
            return point_list_as_seeds([snake.centroid() for snake in snakes], 'snake centroid')

    def find_seeds(self, snakes, all_seeds, exclude_current_segments=False):
        seeds = []

        # Pierwszy krok segmentacji
        if not exclude_current_segments and self.parameters["segmentation"]["seeding"]["from"]["cellBorder"]:
            new_seeds = self.find_seeds_from_border_or_content(self.images.brighter, self.images.foreground_mask,
                                                               self.images.segmentation, 'border',
                                                               exclude_current_segments)
            new_seeds += self.rand_seeds(
                self.random_radius,
                self.parameters["segmentation"]["seeding"]["from"]["cellBorderRandom"],
                new_seeds
            )

            seeds += new_seeds

        # Pierwszy krok segmentacji
        if not exclude_current_segments and self.parameters["segmentation"]["seeding"]["from"]["cellContent"]:
            new_seeds = self.find_seeds_from_border_or_content(self.images.darker, self.images.foreground_mask,
                                                               self.images.segmentation, 'content',
                                                               exclude_current_segments)
            new_seeds += self.rand_seeds(
                self.random_radius,
                self.parameters["segmentation"]["seeding"]["from"]["cellContentRandom"],
                new_seeds
            )

            seeds += new_seeds

        # Jeśli są już jakieś snake'i - todo
        if len(snakes) > 0 and self.parameters["segmentation"]["seeding"]["from"]["cellBorderRemovingCurrSegments"]:
            new_seeds = self.find_seeds_from_border_or_content(self.images.brighter, self.images.foreground_mask,
                                                               self.images.segmentation, 'border', True)
            new_seeds += self.rand_seeds(
                self.random_radius,
                self.parameters["segmentation"]["seeding"]["from"]["cellBorderRemovingCurrSegments"],
                new_seeds
            )

            seeds += new_seeds

        # Jeśli są już jakieś snake'i - todo
        if len(snakes) > 0 and self.parameters["segmentation"]["seeding"]["from"]["cellContentRemovingCurrSegments"]:
            new_seeds = self.find_seeds_from_border_or_content(self.images.darker, self.images.foreground_mask,
                                                               self.images.segmentation, 'content', True)
            new_seeds += self.rand_seeds(
                self.random_radius,
                self.parameters["segmentation"]["seeding"]["from"]["cellContentRemovingCurrSegmentsRandom"],
                new_seeds
            )

            seeds += new_seeds

        # Jeśli są już jakieś snake'i - todo
        if self.parameters["segmentation"]["seeding"]["from"]["snakesCentroids"]:
            new_seeds = self.find_seeds_from_snakes(snakes)
            new_seeds += self.rand_seeds(
                self.random_radius,
                self.parameters["segmentation"]["seeding"]["from"]["snakesCentroidsRandom"],
                new_seeds
            )
            seeds += new_seeds

        #TODO: filter seeds using foreground mask ?
        #seeds = [seed for seed in seeds if self.images.foreground_mask[seed.y, seed.x]]

        seeds = self._filter_seeds(seeds, all_seeds)

        return seeds

    def _filter_seeds(self, seeds, all_seeds):
        """
        @param seeds:
        @type seeds: [contrib.cell_star.core.seed.Seed]
        @return:
        """
        distance = self.parameters["segmentation"]["stars"]["step"] * self.parameters["segmentation"]["avgCellDiameter"]
        distance = float(max(distance, 0.5))   # not less than half of pixel length
        ok_seeds = np.array([False for seed in seeds])

        # TODO: obecnie parametr ustawiony na -1 - wykryć gdzie jest konfigurowany w MATLABIE
        # Wygląda na to, że jest to lista wymiarów klatki/zdjęcia
        # if self.parameters["segmentation"]["transform"]["originalImDim"] > 0:
        grid_size = int(round(max(self.parameters["segmentation"]["transform"]["originalImDim"]) * 1.1 / distance))
        im_x = self.parameters["segmentation"]["transform"]["originalImDim"][1]
        im_y = self.parameters["segmentation"]["transform"]["originalImDim"][0]
        #else:
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
        if euclidean_norm(_seed.as_pair(), seed.as_pair()) < distance:
            return False

    return True


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
