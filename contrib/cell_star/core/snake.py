# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

#external imports
import numpy as np
# from matplotlib.nxutils import points_inside_poly
import math
#internal imports
from contrib.cell_star.utils import calc_util, image_util
from contrib.cell_star.core.point import Point
from contrib.cell_star.utils.index import Index
from contrib.cell_star.utils.python_util import *

class Snake(object):
    epsilon = 0.00000001
    """
    List of contour points
    @ivar x: x coordinate of points
    @ivar y: y coordinate of points
    """

    @classmethod
    def create_from_seed(cls, parameters, seed, points_num, images):
        points = [seed] * points_num
        return cls(parameters, seed, points, None, images)

    @property
    def xs(self):
        return [p.x for p in self.points]

    @property
    def ys(self):
        return [p.y for p in self.points]

    def reinit(self, points=None, final_edgepoints=None):
        self.points = points
        self.final_edgepoints = final_edgepoints
        self.rank = None
        self.max_contiguous_free_border = 0
        self.free_border_entropy = 0.0
        self.in_snake = None
        self.in_polygon = None
        self.in_polygon_yx = None
        self.segment_props = {}
        self.area = 0.0
        self.avg_out_border_brightness = 0.0
        self.max_out_border_brightness = 0.0
        self.avg_in_border_brightness = 0.0
        self.avg_inner_brightness = 0.0
        self.max_inner_brightness = 0.0
        self.avg_inner_darkness = 0.0
        self.centroid_x = 0.0
        self.centroid_y = 0.0
        self.border_integral = 0
        self.polar_coordinate_boundary = None
        self.original_edgepoints = None
        self.properties_vector_cached = {}

    def __init__(self, parameters, seed, points=None, final_edgepoints=None, images=None):
        """
        @type images: core.image_repo.ImageRepo
        @type final_edgepoints: list
        @type points: list
        @type parameters: dict
        @param parameters:
        @param points: 
        @param final_edgepoints: 
        @param images: 
        """
        self.seed = seed
        self.parameters = parameters
        self.points = points
        self.final_edgepoints = final_edgepoints
        self.images = images
        self.rank = None
        self.max_contiguous_free_border = 0
        self.free_border_entropy = 0.0
        self.in_snake = None
        self.in_polygon = None
        self.in_polygon_yx = None
        self.segment_props = {}
        self.area = 0.0
        self.avg_out_border_brightness = 0.0
        self.max_out_border_brightness = 0.0
        self.avg_in_border_brightness = 0.0
        self.avg_inner_brightness = 0.0
        self.max_inner_brightness = 0.0
        self.avg_inner_darkness = 0.0
        self.centroid_x = 0.0
        self.centroid_y = 0.0
        self.border_integral = 0
        self.polar_coordinate_boundary = None
        self.original_edgepoints = None
        self.properties_vector_cached = {}

    def show(self, name):
        maska = image_util.draw_polygons(self.images.image, [zip(self.xs, self.ys)])
        image_out = maska + (1-maska) * self.images.image
        image_util.image_show_and_save(image_out, name, False)

    @speed_profile
    def star_multi_vec(self, size_weight, polar_transform):
        """

        @param size_weight:
        @param polar_transform:
        @type polar_transform: contrib.cell_star.core.vectorized.polar_transform.PolarTransform
        @type seed: contrib.cell_star.core.seed.Seed
        @return:
        """

        #
        #
        # Inicjalizacja
        #
        #

        self.centroid_x = self.seed.x
        self.centroid_y = self.seed.y

        handles = []

        avg_cell_diameter = self.parameters["segmentation"]["avgCellDiameter"]
        points_number = polar_transform.N
        step = polar_transform.step
        unstick = self.parameters["segmentation"]["stars"]["unstick"]
        smoothness = self.parameters["segmentation"]["stars"]["smoothness"]
        gradient_weight = self.parameters["segmentation"]["stars"]["gradientWeight"]
        brightness_weight = self.parameters["segmentation"]["stars"]["brightnessWeight"]
        content_weight = self.parameters["segmentation"]["stars"]["contentWeight"] / avg_cell_diameter
        size_weight = float(size_weight) / avg_cell_diameter
        cum_brightness_weight = self.parameters["segmentation"]["stars"]["cumBrightnessWeight"] / avg_cell_diameter
        background_weight = self.parameters["segmentation"]["stars"]["backgroundWeight"] / avg_cell_diameter

        im = self.images.image_back_difference_blurred
        imb = self.images.brighter
        imc = self.images.darker
        imfg = self.images.foreground_mask

        steps = polar_transform.steps
        max_r = polar_transform.max_r # Taka sama wartość w matlabie - generuje niezgodności !
        R = polar_transform.R.flat
        t = polar_transform.t

        px = float(self.centroid_x) + polar_transform.x
        px = np.maximum(px, 0)
        px = np.minimum(px, im.shape[1] - 1)

        py = float(self.centroid_y) + polar_transform.y
        py = np.maximum(py, 0)
        py = np.minimum(py, im.shape[0] - 1)

        #
        #
        # Obliczanie funkcji oceny konturu i znajdowanie konturu
        #
        #

        # indeks idzie kolejno dla każdego kąta wzdłuż promienia od najmniejszego do największego
        # for angle:
        #   for radius:
        index = calc_util.index(px.round(), py.round()).reshape((polar_transform.x.shape[0], polar_transform.x.shape[1], 2))

        #
        #  pre_f - składowa funkcji oceny jakości konturu
        #
        #
        #

        pre_f = (cum_brightness_weight * imb[Index.to_numpy(index)] - content_weight * imc[Index.to_numpy(index)] + background_weight * (1 - imfg[Index.to_numpy(index)])) * step
        f = np.cumsum(pre_f, axis=0)

        border_thickness_steps = \
            1 + math.floor(float(self.parameters["segmentation"]["stars"]["borderThickness"]) / float(step))

        im_diff = calc_util.get_gradient(im, index, border_thickness_steps)

        f_tot = f \
            - size_weight * np.kron(np.log(R), np.ones((t.size, 1))).T \
            - gradient_weight * im_diff \
            - brightness_weight * im[Index.to_numpy(index)]

        f_tot = f_tot.T


        epsilon = 10**(-10)
        f_tot = (f_tot - f_tot.min()) / (f_tot.max() - f_tot.min() + epsilon)
        f_tot /= (np.kron(np.ones((f_tot.shape[1], 1)), f_tot.max(axis=1)).T + epsilon)

        # Znajdź kontur początkowy
        ymins, xmins = np.where(f_tot == np.kron(np.ones((f_tot.shape[1], 1)), f_tot.min(axis=1)).T)
        # Przytnij - zabezpieczenie przed przekroczeniem dopuszczalnej liczby punktów
        ymins, xmins = ymins[:int(points_number)], xmins[:int(points_number)]

        # image_util.image_show(f_tot, 1)

        #
        #
        # Wygładzanie konturu
        #
        #

        # Posortuj punkty konturu (długości promieni) względem kolejnych kątów
        s = sorted(range(len(ymins)), key=lambda k: ymins[k])
        xmins = xmins[s]
        # Wyznacz maksymalne różnice pomiędzy kolejnymi punktami konturu
        max_diff = np.array(abs(smoothness) * np.arange(1, steps+1) / points_number + 0.5, dtype=int)
        # Pierwsze wygładzenie konturu
        xmins2, xmaxs = self.smooth_contour_vec(xmins, max_diff, points_number, f_tot)

        self.original_edgepoints = (xmins2 != xmaxs) | ((xmins2 == xmins) & (xmins2 < max_r - 2))
        xmins3 = np.minimum(xmins2, max_r)

        # Unstick edgepoint
        # Final edgepoints - punkty minimów, które są brane pod uwagę przy konstruowaniu konturu
        self.final_edgepoints = \
            calc_util.unstick_contour(self.original_edgepoints, self.parameters["segmentation"]["stars"]["unstick"])

        # Interpolate points where no reliable points had been found.
        calc_util.interpolate(self.final_edgepoints, points_number, xmins3)

        #
        # Przetworzenie konturu na listę punktów
        #
        #

        xmins3 += 1

        xmins3 = np.minimum(np.maximum(np.round(xmins3), 1), max_r - 1)

        px = self.seed.x + step * xmins3 * np.cos(t.T)
        py = self.seed.y + step * xmins3 * np.sin(t.T)

        self.polar_coordinate_boundary = xmins3  # np.minimum(np.maximum(np.round(xmins3), 1), max_r - 1)

        # if not np.all(xmins3 == self.polar_coordinate_boundary):
        #     print xmins3
        #     print self.polar_coordinate_boundary
        #
        # assert np.all(xmins3 == self.polar_coordinate_boundary)

        self.points = [Point(x, y) for x, y in zip(np.append(px, px[0]), np.append(py, py[0]))]

        return


    def smooth_contour_vec(self, xmins, max_diff, points_number, f_tot):
        """
        Smoothing contour
        @param xmins: ray lengths for segments
        @type xmins: np.array
        @param max_diff: max change of ray length per iter.
        @type max_diff np.array
        @param points_number: nbr. of points in contour
        @type points_number int
        @param f_tot: energy fun.
        @type f_tot: np.array
        @return:
        """
        points_order = range(0, points_number)
        min_angle = xmins.argmin()
        istart = min_angle

        xmins2 = np.copy(xmins)
        xmaxs = np.copy(xmins)

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
                    xmins2[current] = np.where(f_tot_slice == f_tot_slice.min())[0].max()
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
                    xmins2[current] = max(f_tot[current, :xmaxs[current] + 1].argmin(), xmins2[previous] - max_diff[xmins2[previous]])
                    ok_points = 0
                    changed = True
                else:
                    ok_points -= 1

                current_iteration += 1

            current_iteration += 1

        return xmins2, xmaxs

    def limit_difference(self):
        pass

    def centroid(self):
        return self.centroid_x, self.centroid_y

    @speed_profile
    def calculate_properties_vec(self, polar_transform):
        # Potentially prevent unnecessary calculations
        if self.rank is not None:
            return

        self.properties_vector_cached = {}
        epsilon = 10 ** -8
        min_border_r = 0.055 * self.parameters["segmentation"]["avgCellDiameter"]
        max_border_r = 0.1 * self.parameters["segmentation"]["avgCellDiameter"]
        image_bounds = calc_util.get_cartesian_bounds(self.polar_coordinate_boundary, self.seed.x, self.seed.y, polar_transform)
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
        if self.area > 2*self.epsilon:
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
        self.rank = self.star_rank(self.parameters["segmentation"]["ranking"], self.parameters["segmentation"]["avgCellDiameter"])

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
