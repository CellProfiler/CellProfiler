# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

# External imports
import math
import numpy as np
from image_util import image_show, image_dilate_with_element, get_circle_kernel
from index import Index
from matplotlib.path import Path


def euclidean_norm((x1, y1), (x2, y2)):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


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


def limit_difference(vertices_order, minimal_dist, maximal_dist, max_diff, quality, conservative=False):
    """
    Makes vertices list more 'smooth' by trimming the odd ones. Updates minimal_dist and maximal_dist.
    @param vertices_order: list of sorted angles to consider
    @param conservative: limit change of distance
    @todo what is the purpose of starting over?
    @todo optimize
    """
    vertices_number = len(vertices_order)

    ready = 0
    while ready < vertices_number - 1:
        current = (ready + 1)%vertices_number
        current_angle = vertices_order[current]
        previous = ready%vertices_number
        previous_angle = vertices_order[previous]
        if minimal_dist[current_angle] - minimal_dist[previous_angle] > max_diff[minimal_dist[previous_angle]]:
            maximal_dist[current_angle] = minimal_dist[previous_angle] + max_diff[minimal_dist[previous_angle]]

            # Take quality from 0 to maximal_dist
            # Find best in that interval (can be optimized a lot OPT)
            minimal_dist[current_angle] = np.argmin(quality[:maximal_dist[current_angle]+1,current_angle])
            ready = min(-1,ready - 2) # start over!!

        else:
            ready += 1
    return


def limit_difference_counter(vertices_order, minimal_dist, maximal_dist, max_diff, quality):
    vertices_number = len(vertices_order)

    ready = vertices_number - 1

    while ready > 0:
        current = (ready + 1) % vertices_number
        current_angle = vertices_order[current]
        previous = (ready + 2) % vertices_number
        previous_angle = vertices_order[previous]

        if minimal_dist[current_angle] - minimal_dist[previous_angle] > max_diff[minimal_dist[previous_angle]]:
            maximal_dist[current_angle] = minimal_dist[previous_angle] + max_diff[minimal_dist[previous_angle]]

            minimal_dist[current_angle] = max(
                np.argmin(quality[:maximal_dist[current_angle]+1, current_angle]),
                minimal_dist[previous_angle] - max_diff[minimal_dist[previous_angle]]
            )
            ready = max(vertices_number - 1, ready)
        else:
            ready -= 1

    return minimal_dist, maximal_dist


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
    return np.array(zip(py.flat, px.flat), dtype=np.int64)


def index_decr(px, py):
    return np.array(zip(map(lambda p: int(round(p-1)), py.T.flat), map(lambda p: int(round(p-1)), px.T.flat)))


def get_gradient(im, index, border_thickness_steps):
    """
    Funkcja obliczająca gradient radialny z uwzględnieniem grubości krawędzi komórki
    @param im: obraz, dla którego liczony jest gradient
    @param index: indeksy pikseli obrazka ułożone wg. współrzędnych polarnych (alpha, radius) tj. dla każdej
    współrzędnej kąta(alpha) określającej pojedynczy promień podane są kolejne indeksy pikseli dla kolejnych(rosnących)
    wartości promienia
    @param border_thickness_steps: ilość kroków obliczania gradientu wynikająca z zadanej grubości krawędzi komórki
    @return: macierz gradientu dla zadanego wycinka obrazu wokół zarodka komórki
    """
    # Indeks pomocniczy osi służący do wyznaczenia maksymalnego gradientu
    max_gradient_along_axis = 2
    # Wymiary wycinka obrazu, dla którego będzie obliczany gradient
    radius_lengths, angles = index.shape[0], index.shape[1]
    # Inicjacja macierzy dla obliczania gradientów
    # Dla każdego pojedynczego kroku dla zadanej grubości krawędzi komórki obliczany jest osobny gradient
    # Następnie zwracane są maksymalne wartości gradientu w danym punkcie dla wszystkich kroków grubości krawędzi
    gradients_for_steps = np.zeros((radius_lengths, angles, border_thickness_steps), dtype=np.float64)
    # Dla każdego kroku wynikającego z grubości krawędzi komórki:
    # Najmniejszy krok ma rozmiar 1, największy ma rozmiar: ${border_thickness_steps}
    for border_thickness_step in range(1, int(border_thickness_steps) + 1):

        # Wyznacz początek i koniec wycinka macierzy, dla którego będzie wyliczany gradient
        matrix_end = radius_lengths - border_thickness_step
        matrix_start = border_thickness_step

        # Wyznacz początek i koniec wycinka indeksu pikseli, dla którego będzie wyliczany gradient
        starting_index = index[:matrix_end, :]
        ending_index = index[matrix_start:, :]

        # Wyznacz początek i koniec wycinka macierzy wynikowej, do którego będzie zapisany obliczony gradient
        intersect_start = int(math.ceil(border_thickness_step / 2.0))
        intersect_end = int(intersect_start + matrix_end)

        # Wylicz bieżącą wartość gradientu dla wyznaczonego wycinka obrazu
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


def index_from_polar_transform(polar_transform, centroid_x, centroid_y, image_shape):
    px = float(centroid_x) + polar_transform.x
    px = np.maximum(px, 0)
    px = np.minimum(px, image_shape[1] - 1)

    py = float(centroid_y) + polar_transform.y
    py = np.maximum(py, 0)
    py = np.minimum(py, image_shape[0] - 1)

    # indeks idzie kolejno dla każdego kąta wzdłuż promienia od najmniejszego do największego
    # for angle:
    #   for radius:
    zipped = np.array(zip(py.flat, px.flat), dtype=np.int64)
    zipped = zipped.reshape((polar_transform.x.shape[0], polar_transform.x.shape[1], 2))
    im_index = zipped.reshape(polar_transform.x.size, 2)
    return im_index


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


def polar_to_cartesian(polar_coordinate_boundary, origin_x, origin_y, polar_transform):
    t = polar_transform.t
    step = polar_transform.step
    px = origin_x + step * polar_coordinate_boundary * np.cos(t.T)
    py = origin_y + step * polar_coordinate_boundary * np.sin(t.T)

    return px, py


def star_in_polygon((max_y, max_x), polar_coordinate_boundary, seed_x, seed_y, polar_transform):
    polygon_x, polygon_y = polar_to_cartesian(polar_coordinate_boundary, seed_x, seed_y, polar_transform)

    x1 = max(0, math.floor(min(polygon_x)))
    x2 = min(max_x, math.ceil(max(polygon_x)) + 1)
    y1 = max(0, math.floor(min(polygon_y)))
    y2 = min(max_y, math.ceil(max(polygon_y)) + 1)

    x1 = min(x1, max_x)
    y1 = min(y1, max_y)
    x2 = max(0, x2)
    y2 = max(0, y2)

    small_boolean_mask = get_in_polygon(x1, x2, y1, y2, get_polygon_path(polygon_x, polygon_y))

    boolean_mask = np.zeros((max_y, max_x), dtype=bool)
    boolean_mask[y1:y2, x1:x2] = small_boolean_mask

    xy = [y1, x1]

    return boolean_mask, small_boolean_mask, xy