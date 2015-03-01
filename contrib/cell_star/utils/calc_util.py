# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

# External imports
import math

import numpy as np
from matplotlib.path import Path

from index import Index


def euclidean_norm((x1, y1), (x2, y2)):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def interpolate(final_edgepoints, points_number, xmins3):
    # Interpolacja konturu, na odrzucone punkty
    # Lista indeksów zatwierdzonych punktów konturu
    cumlengths = np.where(final_edgepoints)[0]
    if len(cumlengths) > 0:
        # Dodanie na końcu listy indeksu pierwszego punktu zwiększonego o liczbę
        # punktów konturu, dla obliczenia długości przedziału interpolacji
        cumlengths_loop = np.append(cumlengths, cumlengths[0] + int(points_number))
        for i in range(len(cumlengths)):
            # Indeks bieżącego punktu konturu
            # current = cumlengths[i]
            left_interval_boundary = cumlengths_loop[i]
            # Długość przedziału interpolacji (ilość odrzuconych punktów konturu do najbliższego zatwierdzonego)
            # mlength = cumlengths_loop[i + 1] - current - 1
            interval_length = cumlengths_loop[i + 1] - left_interval_boundary - 1
            # Indeks końca przedziału interpolacji (ostatniego interpolowanego punktu)
            # jend = (current + mlength + 1) % points_number
            right_interval_boundary = cumlengths_loop[i + 1] % points_number

            # Dla każdego punktu w przedziale interpolacji
            for k in range(left_interval_boundary + 1, left_interval_boundary + interval_length + 1):
                # Indeks interpolowanego punktu
                interpolated = k % points_number
                # Oblicz nową interpolowaną wartość
                new_val = round(xmins3[left_interval_boundary] + (
                    xmins3[right_interval_boundary] - xmins3[left_interval_boundary]) * (k - left_interval_boundary) / (
                                    interval_length + 1))
                # Zwróć minimum jako wynik interpolacji - interpolacja nie może oddalić konturu od środka komórki
                xmins3[interpolated] = min(xmins3[interpolated], new_val)


def loop_connected_components(v):
    """
    @param v: numpy.array (1-dim)
    """

    c = []
    init = np.array([])
    fin = []
    if v.sum() > 0:
        c.append(0)
        fin.append(1)
        current = 0
        for i in xrange(0, v.shape[0]):
            if v[i]:
                c[current] += 1
                fin[current] = i
            else:
                if c[current] is not 0:
                    current += 1
                    c.append(0)
                    fin.append(i)

        c = np.array(c)
        fin = np.array(fin)

        if c.shape[0] > 1:
            if c[-1] == 0:
                c = c[0:-1]
                fin = fin[0:-1]

            if v[0] and v[-1]:
                c[0] = c[0] + c[-1]
                c = c[0:-1]
                fin = fin[0:-1]

        init = (fin - c) % v.shape[0] + 1
    return np.array(c), init, fin


def unstick_contour(edgepoints, unstick_coeff):
            """
            Removes edgepoints near previously discarded points.
            @type edgepoints: list of boolean
            @param edgepoints: current edgepoint list
            @type unstick_coeff: float
            @param unstick_coeff
            @return: filtered edgepoints
            """
            (n, init, end) = loop_connected_components(np.logical_not(edgepoints))
            filtered = np.copy(edgepoints)
            n_edgepoint = len(edgepoints)
            for size, s, e in zip(n, init, end):
                for j in range(1,int(size * unstick_coeff + 0.5) + 1):
                    filtered[(e+j) % n_edgepoint] = 0
                    filtered[(s-j) % n_edgepoint] = 0
            return filtered


def sub2ind(dim, (x, y)):
    return x + y * dim


def index(px, py):
    return np.column_stack((py.flat, px.flat)).astype(np.int64)

def get_gradient(im, index, border_thickness_steps):
    """
    Fun. calc. radial gradient including thickness of cell edges
    @param im: image (for which grad. will be calc.)
    @param index: indices of pixes sorted by polar coords. (alpha, radius) 
    @param border_thickness_steps: number of steps to cop. grad. - depands on cell border thickness
    @return: gradient matrix for cell
    """
    # index of axis used to find max grad.
    # PL: Indeks pomocniczy osi służący do wyznaczenia maksymalnego gradientu
    max_gradient_along_axis = 2
    # preparing the image limits (called subimage) for which grad. will be computed
    # PL: Wymiary wycinka obrazu, dla którego będzie obliczany gradient
    radius_lengths, angles = index.shape[0], index.shape[1]
    # matrix init
    # for each single step for each border thick. separated grad. is being computed
    # at the end the max. grad values are returned (for all steps and thick.)
    # PL: Inicjacja macierzy dla obliczania gradientów
    # PL: Dla każdego pojedynczego kroku dla zadanej grubości krawędzi komórki obliczany jest osobny gradient
    # PL: Następnie zwracane są maksymalne wartości gradientu w danym punkcie dla wszystkich kroków grubości krawędzi
    gradients_for_steps = np.zeros((radius_lengths, angles, border_thickness_steps), dtype=np.float64)
    # PL: Dla każdego kroku wynikającego z grubości krawędzi komórki:
    # PL: Najmniejszy krok ma rozmiar 1, największy ma rozmiar: ${border_thickness_steps}
    for border_thickness_step in range(1, int(border_thickness_steps) + 1):

        # find beg. and end indices of input matrix for which the gradient will be computed
        # PL: Wyznacz początek i koniec wycinka macierzy, dla którego będzie wyliczany gradient
        matrix_end = radius_lengths - border_thickness_step
        matrix_start = border_thickness_step

        # find beg. and end indices of pix. for which the gradient will be computed
        # PL: Wyznacz początek i koniec wycinka indeksu pikseli, dla którego będzie wyliczany gradient
        starting_index = index[:matrix_end, :]
        ending_index = index[matrix_start:, :]

        # find the spot in matrix where comp. gradient will go
        # PL: Wyznacz początek i koniec wycinka macierzy wynikowej, do którego będzie zapisany obliczony gradient
        intersect_start = int(math.ceil(border_thickness_step / 2.0))
        intersect_end = int(intersect_start + matrix_end)

        # comp. current gradient for selected (sub)image 
        # PL: Wylicz bieżącą wartość gradientu dla wyznaczonego wycinka obrazu
        try:
            current_step_gradient = im[Index.to_numpy(ending_index)] - im[Index.to_numpy(starting_index)]
        except Exception:
            print border_thickness_step
            print radius_lengths
            print matrix_start
            print matrix_end
            print ending_index
            print starting_index

            raise Exception

        current_step_gradient /= np.sqrt(border_thickness_step)
        # Zapisz gradient do wyznaczonego wycinka macierzy wyników
        gradients_for_steps[intersect_start:intersect_end, :, border_thickness_step-1] = current_step_gradient

    return gradients_for_steps.max(axis=max_gradient_along_axis)


def get_polygon_path(polygon_x, polygon_y):
    vertices = zip(list(polygon_x) + [polygon_x[0]], list(polygon_y) + [polygon_y[0]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
    p = Path(vertices, codes)
    return p


def get_in_polygon(x1, x2, y1, y2, path):
    x, y = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
    x, y = x.flatten(), y.flatten()
    pts = np.vstack((x, y)).T

    # Find points that belong to snake in minimal rectangle
    grid = path.contains_points(pts)
    grid = grid.reshape(y2 - y1, x2 - x1)
    return grid


def inslice_point(point_yx_in_slice, slices):
    y = point_yx_in_slice[0]
    x = point_yx_in_slice[1]
    max_len = 1000000
    return y - slices[0].indices(max_len)[0], x - slices[1].indices(max_len)[0]


def unslice_point(point_yx_in_slice, slices):
    y = point_yx_in_slice[0]
    x = point_yx_in_slice[1]
    max_len = 1000000
    return y + slices[0].indices(max_len)[0], x + slices[1].indices(max_len)[0]


def get_cartesian_bounds(polar_coordinate_boundary, origin_x, origin_y, polar_transform):
    polygon_x, polygon_y = polar_to_cartesian(polar_coordinate_boundary, origin_x, origin_y, polar_transform)
    x1 = int(max(0, math.floor(min(polygon_x))))
    x2 = int(math.ceil(max(polygon_x)) + 1)
    y1 = int(max(0, math.floor(min(polygon_y))))
    y2 = int(math.ceil(max(polygon_y)) + 1)
    return slice(y1, y2), slice(x1, x2)


def polar_to_cartesian(polar_coordinate_boundary, origin_x, origin_y, polar_transform):
    t = polar_transform.t
    step = polar_transform.step
    px = origin_x + step * polar_coordinate_boundary * np.cos(t.T)
    py = origin_y + step * polar_coordinate_boundary * np.sin(t.T)

    return px, py


def star_in_polygon((max_y, max_x), polar_coordinate_boundary, seed_x, seed_y, polar_transform):
    polygon_x, polygon_y = polar_to_cartesian(polar_coordinate_boundary, seed_x, seed_y, polar_transform)

    x1 = int(max(0, math.floor(min(polygon_x))))
    x2 = int(min(max_x, math.ceil(max(polygon_x)) + 1))
    y1 = int(max(0, math.floor(min(polygon_y))))
    y2 = int(min(max_y, math.ceil(max(polygon_y)) + 1))

    x1 = min(x1, max_x)
    y1 = min(y1, max_y)
    x2 = max(0, x2)
    y2 = max(0, y2)

    small_boolean_mask = get_in_polygon(x1, x2, y1, y2, get_polygon_path(polygon_x, polygon_y))

    boolean_mask = np.zeros((max_y, max_x), dtype=bool)
    boolean_mask[y1:y2, x1:x2] = small_boolean_mask

    yx = [y1, x1]

    return boolean_mask, small_boolean_mask, yx