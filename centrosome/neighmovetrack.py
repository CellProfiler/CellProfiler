from __future__ import absolute_import
import math
import copy
import numpy
import scipy.ndimage.measurements
from . import lapjv
import six
from six.moves import range
from six.moves import zip

invalid_match = 1000000  # limiting the choices of the algorithms


def euclidean_dist(point1, point2):
    """Compute the Euclidean distance between two points.

    Parameters
    ----------
    point1, point2 : 2-tuples of float
        The input points.

    Returns
    -------
    d : float
        The distance between the input points.

    Examples
    --------
    >>> point1 = (1.0, 2.0)
    >>> point2 = (4.0, 6.0)  # (3., 4.) away, simplest Pythagorean triangle
    >>> euclidean_dist(point1, point2)
    5.0
    """
    (x1, y1) = point1
    (x2, y2) = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class NeighbourMovementTrackingParameters(object):
    parameters_nbrs = {
        "nbrs_number": 6,
        "nbrs_maxdist": 30
    }

    # max_distance is not scaled by avgCellDiameter as it solely depends on the image motion

    parameters_tracking = {
        "avgCellDiameter": 35,
        "iterations": 20,
        "big_size": 200,
        "max_distance": 300
    }

    parameters_cost_initial = {
        "check_if_big": False,
        "default_empty_cost": 15,
        "default_empty_reliable_cost_mult": 2.5,
        "area_weight": 25
    }

    parameters_cost_iteration = {
        "check_if_big": True,
        "default_empty_cost": 15,
        "default_empty_reliable_cost_mult": 2,
        "area_weight": 30
    }

    parameters_cell_features = {
        "reliable_area": 150,
        "reliable_distance": 30
    }


class CellFeatures(object):
    """
    Represents cell features such as center and its area.

    @ivar number: cell number in the image
    @ivar center: position in the image (row,col)
    @ivar area: area of the cell in pixels
    @ivar image_size: size of the image that feature inhabits (rows,cols)
    """

    def __init__(self, center, area, number, image_size):
        self.parameters = copy.deepcopy(NeighbourMovementTrackingParameters.parameters_cell_features)

        self.center = center

        self.area = area

        self.number = number

        self.image_size = image_size

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(self.number) + ": " + "Center is " + str(self.center) + ", Area is " + str(self.area)

    def distance(self, to_cell):
        return euclidean_dist(self.center, to_cell.center)

    @staticmethod
    def from_labels(labels):
        """
        Creates list of cell features based on label image (1-oo pixel values)

        @return: list of cell features in the same order as labels
        """
        labels = labels.astype(int)

        areas = scipy.ndimage.measurements.sum(labels != 0, labels, list(range(1, numpy.max(labels) + 1)))

        existing_labels = [i for (i, a) in enumerate(areas, 1) if a > 0]

        existing_areas = [a for a in areas if a > 0]

        existing_centers = scipy.ndimage.measurements.center_of_mass(labels != 0, labels, existing_labels)

        zipped = zip(existing_labels, existing_centers, existing_areas)

        features = [CellFeatures(c, a, i, labels.shape)
                    for i, c, a in zipped if a != 0]

        return features

    def is_reliable(self, min_size=-1):
        """

        Determine if detection is considered reliable.

        @return: it is reliable?

        """
        if min_size == -1:
            min_size = self.parameters["reliable_area"]

        return (self.area > min_size and min(self.center[0], min(self.center[1], min(self.image_size[0] - self.center[0], self.image_size[1] - self.center[1]))) > self.parameters["reliable_distance"])


class Trace(object):
    """
    Cell track thorugh time.

    @ivar timepoint: moment in time that represents (not used currently)
    @ivar previous_cell: features of the cell in the previous image
    @ivar current_cell: features of the cell in current image
    @ivar cell_motion: difference in cell position.
    """

    def __init__(self, frame1_cell, frame2_cell):
        self.timepoint = 1

        self.previous_cell = frame1_cell

        self.current_cell = frame2_cell

        self.cell_motion = euclidean_dist(frame1_cell.center, frame2_cell.center)

        self.cell_motion_vector = (frame1_cell.center[0] - frame2_cell.center[0], frame1_cell.center[1] - frame2_cell.center[1])

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "Trace: cells are " + str((self.previous_cell.number, self.current_cell.number)) + " motion is " + str(
            self.cell_motion)

    @staticmethod
    def from_detections_assignment(detections_1, detections_2, assignments):
        """
        Creates traces out of given assignment and cell data.
        """
        traces = []

        for d1n, d2n in six.iteritems(assignments):
            # check if the match is between existing cells
            if d1n < len(detections_1) and d2n < len(detections_2):
                traces.append(Trace(detections_1[d1n], detections_2[d2n]))

        return traces


class NeighbourMovementTracking(object):
    def __init__(self):
        # Copy initial parameters.
        self.parameters_nbrs = copy.deepcopy(NeighbourMovementTrackingParameters.parameters_nbrs)

        self.parameters_tracking = copy.deepcopy(NeighbourMovementTrackingParameters.parameters_tracking)

        self.parameters_cost_initial = copy.deepcopy(NeighbourMovementTrackingParameters.parameters_cost_initial)

        self.parameters_cost_iteration = copy.deepcopy(NeighbourMovementTrackingParameters.parameters_cost_iteration)

        self.parameters_cell_features = copy.deepcopy(NeighbourMovementTrackingParameters.parameters_cell_features)

    def run_tracking(self, label_image_1, label_image_2):
        """
        Tracks cells between input label images.

        @returns: injective function from old objects to new objects (pairs of [old, new]). Number are compatible with labels.
        """
        self.scale = self.parameters_tracking["avgCellDiameter"] / 35.0

        detections_1 = self.derive_detections(label_image_1)

        detections_2 = self.derive_detections(label_image_2)

        # Calculate tracking based on cell features and position.
        traces = self.find_initials_traces(detections_1, detections_2)

        # Use neighbourhoods to improve tracking.
        for _ in range(int(self.parameters_tracking["iterations"])):
            traces = self.improve_traces(detections_1, detections_2, traces)

        # Filter traces.
        return [(trace.previous_cell.number, trace.current_cell.number) for trace in traces]

    def is_cell_big(self, cell_detection):
        """
        Check if the cell is considered big.

        @param CellFeature cell_detection:

        @return:
        """
        return cell_detection.area > self.parameters_tracking["big_size"] * self.scale * self.scale

    @staticmethod
    def derive_detections(label_image):
        """
        Calculate properties for every label/cell.

        List: centroid, area
        """
        return CellFeatures.from_labels(label_image)

    def find_initials_traces(self, detections_1, detections_2):
        # calculate initial costs
        costs = self.calculate_costs(detections_1, detections_2, self.calculate_basic_cost, self.parameters_cost_initial)

        # solve tracking problem
        assignment = self.solve_assignement(costs)

        # create tracks
        traces = Trace.from_detections_assignment(detections_1, detections_2, assignment)

        return traces

    @staticmethod
    def find_closest_neighbours(cell, all_cells, k, max_dist):
        """
        Find k closest neighbours of the given cell.
        :param CellFeatures cell: cell of interest
        :param all_cells: cell to consider as neighbours
        :param int k: number of neighbours to be returned
        :param int max_dist: maximal distance in pixels to consider neighbours
        :return: k closest neighbours
        """
        all_cells = [c for c in all_cells if c != cell]

        sorted_cells = sorted([(cell.distance(c), c) for c in all_cells])

        return [sc[1] for sc in sorted_cells[:k] if sc[0] <= max_dist]

    def calculate_basic_cost(self, d1, d2):
        """
        Calculates assignment cost between two cells.
        """
        distance = euclidean_dist(d1.center, d2.center) / self.scale

        area_change = 1 - min(d1.area, d2.area) / max(d1.area, d2.area)

        return distance + self.parameters_cost_initial["area_weight"] * area_change

    def calculate_localised_cost(self, d1, d2, neighbours, motions):
        """
        Calculates assignment cost between two cells taking into account the movement of cells neighbours.

        :param CellFeatures d1: detection in first frame
        :param CellFeatures d2: detection in second frame
        """
        my_nbrs_with_motion = [n for n in neighbours[d1] if n in motions]

        my_motion = (d1.center[0] - d2.center[0], d1.center[1] - d2.center[1])

        if my_nbrs_with_motion == []:
            distance = euclidean_dist(d1.center, d2.center) / self.scale
        else:
            # it is not in motions if there is no trace (cell is considered to vanish)
            distance = min([euclidean_dist(my_motion, motions[n]) for n in my_nbrs_with_motion]) / self.scale

        area_change = 1 - min(d1.area, d2.area) / max(d1.area, d2.area)

        return distance + self.parameters_cost_iteration["area_weight"] * area_change

    def calculate_costs(self, detections_1, detections_2, calculate_match_cost, params):
        """
        Calculates assignment costs between detections and 'empty' spaces. The smaller cost the better.

        @param detections_1: cell list of size n in previous frame
        @param detections_2: cell list of size m in current frame

        @return: cost matrix (n+m)x(n+m) extended by cost of matching cells with emptiness
        """
        global invalid_match

        size_sum = len(detections_1) + len(detections_2)

        # Cost matrix extended by matching cells with nothing
        # (for detection 1 it means losing cells, for detection 2 it means new cells).
        cost_matrix = numpy.zeros((size_sum, size_sum))

        # lost cells cost
        cost_matrix[0:len(detections_1), len(detections_2):size_sum] = params["default_empty_cost"] + (1 - numpy.eye(
            len(detections_1), len(detections_1))) * invalid_match

        # new cells cost
        cost_matrix[len(detections_1):size_sum, 0:len(detections_2)] = params["default_empty_cost"] + (1 - numpy.eye(
            len(detections_2), len(detections_2))) * invalid_match

        # increase costs for reliable detections
        for row in [i for i in range(0, len(detections_1)) if detections_1[i].is_reliable() and (
            not params["check_if_big"] or self.is_cell_big(detections_1[i]))]:
            cost_matrix[row, len(detections_2):size_sum] *= params["default_empty_reliable_cost_mult"]

        for col in [i for i in range(0, len(detections_2)) if detections_2[i].is_reliable() and (
            not params["check_if_big"] or self.is_cell_big(detections_2[i]))]:
            cost_matrix[len(detections_1):size_sum, col] *= params["default_empty_reliable_cost_mult"]

        # calculate cost of matching cells
        def cost_if_not_too_far(detection_1, detection_2):
            if detection_1.distance(detection_2) <= self.parameters_tracking["max_distance"]:
                return calculate_match_cost(detection_1, detection_2)
            else:
                return invalid_match

        cost_matrix[0:len(detections_1), 0:len(detections_2)] = [[cost_if_not_too_far(d1, d2) for d2 in detections_2] for d1 in detections_1]

        return cost_matrix

    def improve_traces(self, detections_1, detections_2, traces):
        # calculate cell motion and neighbours
        cells_motion = dict([(t.previous_cell, t.cell_motion_vector) for t in traces])

        neighbours = dict([(d, NeighbourMovementTracking.find_closest_neighbours(d, detections_1, self.parameters_nbrs["nbrs_number"], self.parameters_nbrs["nbrs_maxdist"] * self.scale)) for d in detections_1])

        # calculate localised costs
        cost_function = lambda d1, d2: self.calculate_localised_cost(d1, d2, neighbours, cells_motion)

        localized_costs = self.calculate_costs(detections_1, detections_2, cost_function, self.parameters_cost_iteration)

        # solve tracking problem

        assignment = self.solve_assignement(localized_costs)

        # create tracks
        improved_traces = Trace.from_detections_assignment(detections_1, detections_2, assignment)

        return improved_traces

    def solve_assignement(self, costs):
        """
        Solves assignment problem using Hungarian implementation by Brian M. Clapper.

        @param costs: square cost matrix

        @return: assignment function

        @rtype: int->int
        """
        if costs is None or len(costs) == 0:
            return dict()

        n = costs.shape[0]

        pairs = [(i, j) for i in range(0, n) for j in range(0, n) if costs[i, j] < invalid_match]

        costs_list = [costs[i, j] for (i, j) in pairs]

        assignment = lapjv.lapjv(list(zip(*pairs))[0], list(zip(*pairs))[1], costs_list)

        indexes = enumerate(list(assignment[0]))

        return dict([(row, col) for row, col in indexes])
