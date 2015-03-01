# -*- coding: utf-8 -*-
"""
Snake object responsible for growing best contours from given seed.
Date: 2013-2016
Website: http://cellstar-algorithm.org/
"""

import math

import numpy as np

from contrib.cell_star.core.point import Point
from contrib.cell_star.utils import calc_util, image_util
from contrib.cell_star.utils.index import Index
from contrib.cell_star.utils.python_util import *


class Snake(object):
    """
    Contour representation of snake along with all its properties.
    @ivar in_polygon : cropped array representation of contour
    @ivar in_polygon_yx : position of in_polygon
    @ivar rank: ranking of the contour (the smaller the better)
    """
    epsilon = 1e-10

    @property
    def xs(self):
        return [p.x for p in self.points]

    @property
    def ys(self):
        return [p.y for p in self.points]

    @property
    def centroid(self):
        return self.centroid_x, self.centroid_y

    @property
    def in_polygon_slice(self):
        if self.in_polygon_yx is None or self.in_polygon is None:
            return None

        yx = self.in_polygon_yx
        local = self.in_polygon
        return slice(yx[0], yx[0] + local.shape[0]), slice(yx[1], yx[1] + local.shape[1])

    @classmethod
    def create_from_seed(cls, parameters, seed, points_num, images):
        points = [seed] * points_num
        return cls(parameters, seed, points, images)

    def __init__(self, parameters, seed, points=None, images=None):
        """
        @type seed: Seed
        @type images: core.image_repo.ImageRepo
        @type points: list[Point]
        @type parameters: dict
        """
        self.seed = seed
        self.parameters = parameters
        self.images = images

        # Snake grow.
        self.points = points
        self.original_edgepoints = None
        self.final_edgepoints = None
        self.polar_coordinate_boundary = None

        # Snake evaluation properties.
        self.in_polygon = None
        self.in_polygon_yx = None
        self.rank = None
        self.area = 0.0
        self.avg_out_border_brightness = 0.0
        self.max_out_border_brightness = 0.0
        self.avg_in_border_brightness = 0.0
        self.avg_inner_brightness = 0.0
        self.max_inner_brightness = 0.0
        self.avg_inner_darkness = 0.0
        self.centroid_x = 0.0
        self.centroid_y = 0.0
        self.max_contiguous_free_border = 0
        self.free_border_entropy = 0.0
        self.properties_vector_cached = {}

    @speed_profile
    def star_grow(self, size_weight, polar_transform):
        """
        Grow the snake from seed.
        @type polar_transform: contrib.cell_star.core.vectorized.polar_transform.PolarTransform
        @type size_weight: float
        """

        #
        # Initialization
        #

        self.centroid_x = self.seed.x
        self.centroid_y = self.seed.y

        points_number = polar_transform.N
        step = polar_transform.step
        unstick = self.parameters["segmentation"]["stars"]["unstick"]
        avg_cell_diameter = self.parameters["segmentation"]["avgCellDiameter"]
        smoothness = self.parameters["segmentation"]["stars"]["smoothness"]
        gradient_weight = self.parameters["segmentation"]["stars"]["gradientWeight"]
        brightness_weight = self.parameters["segmentation"]["stars"]["brightnessWeight"]
        content_weight = self.parameters["segmentation"]["stars"]["contentWeight"] / avg_cell_diameter
        size_weight = float(size_weight) / avg_cell_diameter
        cum_brightness_weight = self.parameters["segmentation"]["stars"]["cumBrightnessWeight"] / avg_cell_diameter
        background_weight = self.parameters["segmentation"]["stars"]["backgroundWeight"] / avg_cell_diameter
        border_thickness_steps = \
            1 + math.floor(float(self.parameters["segmentation"]["stars"]["borderThickness"]) / float(step))

        im = self.images.image_back_difference_blurred
        imb = self.images.brighter
        imc = self.images.darker
        imfg = self.images.foreground_mask

        steps = polar_transform.steps
        max_r = polar_transform.max_r
        R = polar_transform.R.flat
        t = polar_transform.t
        max_diff = (abs(smoothness) * np.arange(1, steps + 1) / points_number + 0.5).astype(int)

        px = float(self.centroid_x) + polar_transform.x
        px = np.maximum(px, 0)
        px = np.minimum(px, im.shape[1] - 1)

        py = float(self.centroid_y) + polar_transform.y
        py = np.maximum(py, 0)
        py = np.minimum(py, im.shape[0] - 1)

        #
        # Contour quality function calculation and finding the best candidates.
        #
        #
        # index is ordered first by angle then by radius
        # for angle:
        #   for radius:

        index = calc_util.index(px.round(), py.round()).reshape(
            (polar_transform.x.shape[0], polar_transform.x.shape[1], 2))

        #
        #  pre_f - part of function quality for interior
        #  f_tot - final quality array
        #

        numpy_index = Index.to_numpy(index)
        pre_f = (cum_brightness_weight * imb[numpy_index] \
                 - content_weight * imc[numpy_index] \
                 + background_weight * (1 - imfg[numpy_index])) * step
        f = np.cumsum(pre_f, axis=0)

        im_diff = calc_util.get_gradient(im, index, border_thickness_steps)

        f_tot = f \
                - size_weight * np.kron(np.log(R), np.ones((t.size, 1))).T \
                - gradient_weight * im_diff \
                - brightness_weight * im[numpy_index]

        f_tot = f_tot.T
        # f_tot = image_util.image_smooth(f_tot, 1)

        # Scale entire array to 0-1 then scale individual angles.
        f_tot = (f_tot - f_tot.min()) / (f_tot.max() - f_tot.min() + self.epsilon)
        f_tot /= (np.kron(np.ones((f_tot.shape[1], 1)), f_tot.max(axis=1)).T + self.epsilon)

        # Find initial contour
        _, best_radius = np.where(f_tot == np.kron(np.ones((f_tot.shape[1], 1)), f_tot.min(axis=1)).T)

        #
        # Contour smoothing
        #

        smoothed_radius, radius_bounds = self.smooth_contour(best_radius, max_diff, points_number, f_tot)

        self.original_edgepoints = (smoothed_radius != radius_bounds) | \
                                   ((smoothed_radius == best_radius) & (smoothed_radius < max_r - 2))
        smoothed_radius = np.minimum(smoothed_radius, max_r)

        # Determine final edgepoint that will be used to construct contour points
        self.final_edgepoints = \
            calc_util.unstick_contour(self.original_edgepoints, unstick)

        # Interpolate points where no reliable points had been found.
        calc_util.interpolate(self.final_edgepoints, points_number, smoothed_radius)

        #
        # Create contour points list
        #

        final_radius = np.minimum(np.maximum(np.round(smoothed_radius + 1), 1), max_r - 1)

        px = self.seed.x + step * final_radius * np.cos(t.T)
        py = self.seed.y + step * final_radius * np.sin(t.T)

        self.polar_coordinate_boundary = final_radius
        self.points = [Point(x, y) for x, y in zip(px, py)]

    def smooth_contour(self, radius, max_diff, points_number, f_tot):
        """
        Smoothing contour using greedy length cut. Rotating from min radius clockwise and anti.
        @type radius: np.ndarray
        @param max_diff: max change of ray length per iter.
        @type max_diff np.ndarray
        @type points_number int
        @param f_tot: quality function array
        @type f_tot: np.ndarray

        @rtype (np.ndarray, np.ndarray)
        @return (smoothed_radius, used_radius_bounds)
        """
        points_order = range(0, points_number)
        min_angle = radius.argmin()
        istart = min_angle

        xmins2 = np.copy(radius)
        xmaxs = np.copy(radius)

        current_iteration = 0
        ok_points = 0
        changed = True

        while changed:
            changed = False

            # vertices_order = points_order[min_angle:] + points_order[:min_angle]
            fixed = 0
            while ok_points < points_number:
                if fixed >= points_number and ok_points != 0:
                    current_iteration += points_number - ok_points
                    ok_points = points_number
                    break

                current = (istart + current_iteration) % points_number
                previous = (current - 1) % points_number

                if xmins2[current] - xmins2[previous] > max_diff[xmins2[previous]]:
                    xmaxs[current] = xmins2[previous] + max_diff[xmins2[previous]]
                    f_tot_slice = f_tot[current, :xmaxs[current] + 1]
                    xmins2[current] = f_tot_slice.argmin()
                    ok_points = 0
                    changed = True
                else:
                    ok_points += 1

                fixed += 1
                current_iteration += 1

            while ok_points > 1:
                current = (istart + current_iteration) % points_number
                previous = (current + 1) % points_number

                if xmins2[current] - xmins2[previous] > max_diff[xmins2[previous]]:
                    xmaxs[current] = xmins2[previous] + max_diff[xmins2[previous]]
                    f_tot_slice = f_tot[current, :xmaxs[current] + 1]
                    xmins2[current] = max(f_tot_slice.argmin(), xmins2[previous] - max_diff[xmins2[previous]])
                    ok_points = 0
                    changed = True
                else:
                    ok_points -= 1

                current_iteration += 1

            current_iteration += 1

        return xmins2, xmaxs

    @speed_profile
    def calculate_properties_vec(self, polar_transform):
        """
        Analyse contour and calculate all it properties and ranking.
        @type polar_transform: contrib.cell_star.core.vectorized.polar_transform.PolarTransform
        """

        # Potentially prevent unnecessary calculations
        if self.rank is not None:
            return

        self.properties_vector_cached = {}

        avg_cell_diameter = self.parameters["segmentation"]["avgCellDiameter"]
        min_border_r = 0.055 * avg_cell_diameter
        max_border_r = 0.1 * avg_cell_diameter
        ranking_params = self.parameters["segmentation"]["ranking"]

        image_bounds = calc_util.get_cartesian_bounds(self.polar_coordinate_boundary, self.seed.x, self.seed.y,
                                                      polar_transform)
        dilated_bounds = image_util.extend_slices(image_bounds, int(max_border_r * 2))

        original_clean = self.images.image_back_difference[dilated_bounds]
        brighter = self.images.brighter[dilated_bounds]
        cell_content_mask = self.images.cell_content_mask[dilated_bounds]
        origin_in_slice = calc_util.inslice_point((self.seed.y, self.seed.x), dilated_bounds)
        origin_in_slice = Point(x=origin_in_slice[1], y=origin_in_slice[0])

        segment, self.in_polygon, in_polygon_local_yx = calc_util.star_in_polygon(brighter.shape,
                                                                                  self.polar_coordinate_boundary,
                                                                                  origin_in_slice.x, origin_in_slice.y,
                                                                                  polar_transform)
        self.in_polygon_yx = calc_util.unslice_point(in_polygon_local_yx, dilated_bounds)

        self.area = np.count_nonzero(self.in_polygon) + self.epsilon
        approx_radius = math.sqrt(self.area / math.pi)
        border_radius = max(min(approx_radius, max_border_r), min_border_r)

        dilation = round(border_radius / polar_transform.step)

        dilated_boundary = np.minimum(self.polar_coordinate_boundary + dilation, len(polar_transform.R) - 1)
        eroded_boundary = np.maximum(self.polar_coordinate_boundary - dilation, 1)

        dilated, _, _ = calc_util.star_in_polygon(brighter.shape, dilated_boundary,
                                                  origin_in_slice.x, origin_in_slice.y, polar_transform)

        eroded, _, _ = calc_util.star_in_polygon(brighter.shape, eroded_boundary,
                                                 origin_in_slice.x, origin_in_slice.y, polar_transform)

        out_border = dilated ^ segment
        in_border = segment ^ eroded

        out_border_area = np.count_nonzero(out_border) + self.epsilon
        in_border_area = np.count_nonzero(in_border) + self.epsilon

        # Calculate outer border brightness
        tmp = original_clean[out_border]
        self.avg_out_border_brightness = tmp.sum() / out_border_area
        self.max_out_border_brightness = tmp.max() if tmp != [] else 0

        # Calculate inner border brightness
        tmp = original_clean[in_border]
        self.avg_in_border_brightness = tmp.sum() / in_border_area

        # Calculate inner brightness
        tmp = brighter[segment]
        self.avg_inner_brightness = tmp.sum() / self.area
        self.max_inner_brightness = tmp.max() if tmp != [] else 0

        # Calculate inner darkness
        tmp = cell_content_mask[segment]
        self.avg_inner_darkness = tmp.sum() / self.area

        # Calculate snake centroid
        if self.area > 2 * self.epsilon:
            area2 = float(self.area - self.epsilon)
            self.centroid_x = (self.in_polygon.sum(0) * np.arange(1, self.in_polygon.shape[1] + 1)).sum() / area2
            self.centroid_y = (self.in_polygon.sum(1) * np.arange(1, self.in_polygon.shape[0] + 1)).sum() / area2
            self.centroid_x += self.in_polygon_yx[1]
            self.centroid_y += self.in_polygon_yx[0]
        else:
            self.centroid_x = self.xs[0]
            self.centroid_y = self.ys[0]

        # Calculate free border fragments - fragments of border, which endpoints have been discarded
        fb, _, _ = calc_util.loop_connected_components(np.logical_not(self.final_edgepoints))

        # Calculate free border entropy
        fb_entropy = 0
        if min(fb.shape) != 0:
            fb_entropy = float(np.sum(fb * fb.T)) / len(self.xs) ** 2

        # The longest continuous fragment of free border
        self.max_contiguous_free_border = fb.max() if fb.size > 0 else 0

        self.free_border_entropy = fb_entropy
        self.rank = self.star_rank(ranking_params, avg_cell_diameter)

    def star_rank(self, ranking_params, avg_cell_diameter):
        return np.dot(self.ranking_parameters_vector(ranking_params), self.properties_vector(avg_cell_diameter))

    def ranking_parameters_vector(self, ranking_params):
        return np.array([
            float(ranking_params["maxInnerBrightnessWeight"]),
            float(ranking_params["avgInnerBrightnessWeight"]),
            float(ranking_params["avgBorderBrightnessWeight"]),
            float(ranking_params["avgInnerDarknessWeight"]),
            float(ranking_params["logAreaBonus"]),
            float(ranking_params["stickingWeight"]),
        ])

    def properties_vector(self, avg_cell_diameter):
        if avg_cell_diameter not in self.properties_vector_cached:
            self.properties_vector_cached[avg_cell_diameter] = np.array([
                self.max_inner_brightness,
                self.avg_inner_brightness,
                self.avg_in_border_brightness - self.avg_out_border_brightness,
                -self.avg_inner_darkness,
                -math.log(self.area ** (1.0 / avg_cell_diameter)),
                self.free_border_entropy
            ])
        return self.properties_vector_cached[avg_cell_diameter]
