# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

# External imports
import numpy as np
import math
import threading
# Internal imports
from contrib.cell_star.utils.image_util import image_dilate_with_element, get_circle_kernel, image_show
from contrib.cell_star.utils.calc_util import sub2ind


class PolarTransform(object):
    """
    Object wrapping polar transform calculations and cached properties

    @type N: int
    @ivar N: number of points for polar transform calculation

    @type distance: float
    @ivar distance: maximal distance from cell center to cell border

    @type step: float
    @ivar step: length of step for active contour along its axis expressed in pixels

    @type steps: int
    @ivar steps: number of steps considered for active contour along single axis

    @type max_r: int
    @ivar max_r: maximal radius of active contour

    @type R: numpy.array
    @ivar R: consecutive radii values for single axis of active contour

    @type center: int
    @ivar center: Polar transform center coordinate

    @type edge: int
    @ivar edge: dimension of polar transform

    @type halfedge: int
    @ivar halfedge: half of polar transform dimension

    @type t: numpy.array
    @ivar t: angles (in radians) of consecutive rays casted from 'center'

    @type x: numpy.array
    @ivar x: cartesian x-coordinates of points in polar coordinates system
    coordinates ordered by radius of polar points --> x[r,a] = P(r,a).x

    @type y: numpy.array
    @ivar y:cartesian y-coordinates of points in polar coordinates system
    coordinates ordered by radius of polar points --> y[r,a] = P(r,a).y

    @type dot_voronoi: numpy.array
    @ivar dot_voronoi - diagram voronoi'a - "pole przyciągania" poszczególnych punktów konturu
    dot_voronoi[x,y] = id(closest_contour_point(x,y))


    @type to_polar: dict
    @ivar to_polar - słownik list - dla każdego punktu
    to_polar[index(P(R,a)] - lista indeksów punktów w obszarze przyciągania punktów konturu {P(r,a)| 0 < r < R}
    to_polar[index(P(R,a)] = [gravity_field(dot_voronoi, p) for p in {P(r,a) | 0 < r < R}]
    to_polar[index(P(R,a)] =
    [index(x,y) for x,y in range((0,0),(edge,edge)) if dot_voronoi[x,y] == gravity_index(p) for p in {P(r,a) | 0 < r < R}]
    """

    __singleton_lock = threading.Lock()
    __singleton_instances = {}

    @classmethod
    def instance(cls, avg_cell_diameter, points, step, max_size):
        init_params = avg_cell_diameter, points, step, max_size
        if not cls.__singleton_instances.get(init_params, False):
            with cls.__singleton_lock:
                if not cls.__singleton_instances.get(init_params, False):
                    cls.__singleton_instances[init_params] = cls(avg_cell_diameter, points, step, max_size)
        return cls.__singleton_instances.get(init_params, None)

    def __init__(self, avg_cell_diameter, points, step, max_size):

        self.N = points
        self.distance = max_size * avg_cell_diameter
        self.step = max(step * avg_cell_diameter, 0.2)
        self.steps = 1 + int(round((self.distance + 2) / self.step))
        self.max_r = min(1 + int(round(self.distance / self.step)), self.steps - 1)

        self.R = None
        self.center = None
        self.edge = None
        self.half_edge = None
        self.x = None
        self.y = None

        self.dot_voronoi = None
        self.to_polar = {}
        # self.to_polar_np

        self._calculate_polar_transform()

    def _calculate_polar_transform(self):
        self.R = np.arange(1, self.steps+1).reshape((1, self.steps)).transpose() * self.step

        # Kolejne kąty promieni rzucanych ze środka komórki
        self.t = np.linspace(0, 2 * math.pi, self.N + 1)
        self.t = self.t[:-1]

        # sinusy i cosinusy kolejnych kątów - powtórzone steps-krotnie
        # wartości funkcji dla kątów w każdej odległości od środka (dla danego kąta takie same dla wszystkich odległości)
        sin_t = np.kron(np.ones((len(self.R), 1)), np.sin(self.t))
        cos_t = np.kron(np.ones((len(self.R), 1)), np.cos(self.t))

        # N-krotnie powtórzony wektor kolejnych długości promienia
        RR = np.kron(np.ones((1, len(self.t))), self.R)

        # Z definicji współrzędnych biegunowych:
        # x - macierz współrzędnych x-owych dla kąta \alpha i długości promienia R
        # y - macierz współrzędnych y-owych dla kąta \alpha i długości promienia R
        self.x = RR * cos_t
        self.y = RR * sin_t

        self.half_edge = math.ceil(self.R[-1] + 2)
        self.center = self.half_edge + 1
        self.edge = self.center + self.half_edge

        # Czysty (czarny) obraz o wymiarach edge x edge
        self.dot_voronoi = np.zeros((self.edge, self.edge))
        px = self.center + self.x
        py = self.center + self.y

        # Utwórz listę par współrzędnych (x,y) punktów na sprawdzanym konturze
        index = np.column_stack(((py - .5).astype(int).T.flat, (px - .5).astype(int).T.flat))

        # Utwórz listę kolejnych identyfikatorów dla w/w punktów
        cont = np.arange(1, px.size+1)

        # Zaznacz na obrazie 'dot_voronoi' każdy z rozważanych punktów użwając unikalnego identyfikatora dla każdego punktu
        self.dot_voronoi[tuple(index.T)] = cont

        # W kolejnych iteracjach "rozmywaj" obraz zaznaczając "pole" przyciągania poszczególnych punktów
        # vide. diagramy voronoi
        for i in range(0, int(self.center)):
            ndv = image_dilate_with_element(self.dot_voronoi, 3)
            mask = np.logical_and((self.dot_voronoi == 0), (ndv != 0))
            mask = mask.nonzero()
            self.dot_voronoi[mask] = ndv[mask]

        # Na uzyskany diagram voronoi'a zaaplikuj maskę o kształcie koła
        circ_mask = get_circle_kernel(self.half_edge)
        self.dot_voronoi[np.logical_not(circ_mask)] = 0
        self.dot_voronoi[self.center - 1, self.center - 1] = 0

        # self.to_polar_np = np.empty((self.t.size * self.R.size,), dtype=object)
        # Dla każdego kąta
        for a in range(self.t.size):
            # Utwórz nową maskę - czystą (same zera)
            mask = np.zeros((self.edge, self.edge),dtype=bool)
            # Dla każdego promienia
            for r in range(self.R.size):
                # Znajdź indeks punktu P(r,a)
                idx = sub2ind(px.shape[0], (r, a))
                val = idx + 1
                # Znajdź punkty na 'dot_voronoi' o wartości indeksu
                indices = np.array(zip(*np.nonzero(self.dot_voronoi == val)))
                # Ustaw maskę na 1 w w/w punktach
                if len(indices) > 0:
                    mask[tuple(indices.T)] = 1

                # Wartość to_polar[idx] jest listą par współrzędnych (x,y) punktów na masce
                self.to_polar[idx] = map(lambda pair: pair, zip(*np.nonzero(mask)))
                # self.to_polar_np[idx] = map(lambda pair: pair, zip(*np.nonzero(mask)))