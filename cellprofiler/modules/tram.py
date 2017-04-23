from cellprofiler.modules import trackobjects

CAT_TRAM = "TrAM"
MEAS_TRAM = "TrAM"
FULL_TRAM_MEAS_NAME = "%s_%s" % (CAT_TRAM, MEAS_TRAM)
IMAGE = "Image"
MIN_TRAM_LENGTH = 6 # minimum number of timepoints to calculate TrAM

# todo
__doc__ = ""

import logging

logger = logging.getLogger(__name__)

import numpy as np
import itertools as it
import cellprofiler.module as cpm
import cellprofiler.setting as cps
import cellprofiler.measurement as cpmeas
from collections import Counter
import re

class TrAM(cpm.Module):
    module_name = "TrAM"
    category = "Object Processing"
    variable_revision_number = 1

    def create_settings(self):
        # for them to choose the tracked objects
        self.object_name = cps.ObjectNameSubscriber(
                "Select the tracked objects", cps.NONE, doc="""
            Select the tracked objects for computing TrAM.""")

        # which measurements will go into the TrAM computation todo: possible to restrict measurements shown?
        self.tram_measurements = cps.MeasurementMultiChoice(
            "TrAM measurements", doc="""
            This setting defines the tracked quantities that will be used
            to compute the TrAM metric. At least one must be selected.""")

        self.wants_XY_Euclidian = cps.Binary(
            'Euclidian XY metric?', True, doc='''
            Euclidianize the metric for all measurements occurring in X-Y pairs
            ''')

        # spline knots
        self.num_knots = cps.Integer(
            "Number of spline knots", 4, minval=3, doc="""
            Number of knots to use in the spline fit to time series""")

        # TrAM exponent
        self.p = cps.Float(
            "TrAM exponent", 0.5, minval=0.01, maxval=1, doc="""
            The exponent used to combine different measurements.""")

    def settings(self):
        return [self.object_name, self.tram_measurements, self.wants_XY_Euclidian, self.num_knots, self.p]

    def validate_module(self, pipeline):
        '''Make sure that the user has selected at least one measurement for TrAM'''
        if (len(self.get_selected_tram_measurements()) == 0):
            raise cps.ValidationError(
                    "Please select at least one TrAM measurement for tracking of " + self.object_name.get_value(),
                    self.tram_measurements)

    def visible_settings(self):
        return self.settings()

    def run(self, workspace):
        """Run the module (abstract method)

        workspace    - The workspace contains
            pipeline     - instance of cpp for this run
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - the parent frame to whatever frame is created.
                           None means don't draw.
            display_data - the run() module should store anything to be
                           displayed in this attribute, which will be used in
                           display()

        run() should not attempt to display any data, but should communicate it
        to display() via the workspace.
        """
        pass



    def display_post_group(self, workspace, figure):
        """Display the results of work done post-group

        This method is only called if self.show_window is True

        workspace - the current workspace. workspace.display_data should have
                    whatever information is needed for the display. Numpy arrays
                    lists, tuples, dictionaries and Python builtin objects are
                    allowed.

        figure - the figure to use for the display.
        """
        # get the TrAM values. Just for the first time point because they are just duplicadted across time
        if self.show_window:
            figure.set_subplots((1, 1))
            figure.subplot_histogram(0, 0,
                                     workspace.display_data.tram_values,
                                     bins=50,
                                     xlabel="TrAM",
                                     ylabel="Count",
                                     title="TrAM for %s" % self.object_name.get_value())
            figure.Refresh()


    def post_group(self, workspace, grouping):
        self.show_window = True

        measurements = workspace.measurements
        obj_name = self.object_name.get_value()
        obj_names = measurements.get_object_names()
        img_numbers = measurements.get_image_numbers()
        num_images = len(img_numbers)

        # get vector of tracking label for each data point
        feature_names_list = [measurements.get_feature_names(obj_name) for obj_name in obj_names]# todo: make sure obj_names is of length 1?
        feature_names = dict(zip(obj_names, feature_names_list))[obj_name]
        tracking_label_feature_name = [name for name in feature_names
                                       if name.startswith(trackobjects.F_PREFIX + "_" + trackobjects.F_LABEL)][0]
        label_vals = measurements.get_measurement(obj_name, tracking_label_feature_name, img_numbers)
        label_dict = {tracking_label_feature_name : label_vals} # keep it

        # get vector of lifetime for each tracking point
        lifetime_feature_name = [name for name in feature_names
                                 if name.startswith(trackobjects.F_PREFIX + "_" + trackobjects.F_LIFETIME)][0]
        lifetime_vals = measurements.get_measurement(obj_name, lifetime_feature_name, img_numbers)

        selections = self.get_selected_tram_measurements()

        # get a tuple dictionary entry relating feature name with data values
        def get_dict_entry(sel):
            feat_obj_name, feat_name = sel.split("|")
            vals = measurements.get_measurement(feat_obj_name, feat_name, measurements.get_image_numbers())
            return (feat_name, vals)

        dict_entry_list = [get_dict_entry(sel) for sel in selections]
        all_values_dict = dict(dict_entry_list) # contains dictionary from feature name to values
        # determine if there are any euclidian (XY) pairs
        if self.wants_XY_Euclidian.value:
            euclidian_pairs = TrAM.Determine_Euclidian_pairs(all_values_dict.keys())
        else:
            euclidian_pairs = []

        all_values_dict.update(label_dict) # add the label dictionary

        # get image numbers into the dict
        counts = [len(x) for x in label_vals] # number of tracked objects at each time point
        z = zip(img_numbers, counts)
        image_vals = [[image for _ in xrange(count)] for image, count in zip(img_numbers, counts)]
        all_values_dict.update({IMAGE : image_vals})

        # We have all the data in lists of lists. Flatten.
        def flatten_list_of_lists(lol):
            return list(it.chain.from_iterable(lol))
        vector_dict = {k : flatten_list_of_lists(v) for k, v in all_values_dict.items()}

        lifetime_vals_flattened = flatten_list_of_lists(lifetime_vals)
        label_vals_flattened = flatten_list_of_lists(label_vals)
        max_lifetime_by_label = dict(max(lifetimes)
                                     for label, lifetimes
                                     in it.groupby(zip(label_vals_flattened, lifetime_vals_flattened),
                                                   lambda x: x[0]))


        # Include only labels which appear exactly num_images times and have max lifetime of num_images.
        # These are the cells that are tracked the whole time. todo: figure out a way to deal with mitosis
        label_counts = Counter(label_vals_flattened) # dict with count of each label
        labels_for_complete_trajectories = [label for label in max_lifetime_by_label.keys()
                                            if max_lifetime_by_label[label] == num_images
                                            and label_counts[label] == num_images]

        # create dictionary that doesn't have the tracking label or image number since these will never be TrAMmed.
        data_only_dict = vector_dict.copy()
        data_only_dict.pop(tracking_label_feature_name) # remove
        image_vals_flattened = data_only_dict.pop(IMAGE) # remove and keep

        # now restrict vectors only to labels of complete trajectories
        keep_indices = [i for i, label in enumerate(label_vals_flattened) if label in labels_for_complete_trajectories]
        data_only_dict_complete_trajectories = {k : [v[i] for i in keep_indices] for k, v in data_only_dict.items()}

        # compute typical inter-timepoint variation
        label_vals_flattened_complete_trajectories =[label_vals_flattened[i] for i in keep_indices]
        image_vals_flattened_complete_trajectories = [image_vals_flattened[i] for i in keep_indices]
        tad = TrAM.compute_typical_deviations(data_only_dict_complete_trajectories,
                                              label_vals_flattened_complete_trajectories,
                                              image_vals_flattened_complete_trajectories)

        # put all the data into a 2D array and normalize by typical deviations
        data_array = np.column_stack(data_only_dict_complete_trajectories.values())
        data_keys = data_only_dict_complete_trajectories.keys()
        inv_devs = np.diag([1/tad[k] for k in data_keys]) # inverse of typical deviation
        normalized_data_array = np.dot(data_array, inv_devs) # perform the multiplication

        # now get all unique labels and run through them, computing tram
        tram_dict = dict()
        for label in set(label_vals_flattened):
            # if not a complete trajectory then the answer is nan
            if label not in labels_for_complete_trajectories:
                tram = float('nan')
            else:
                indices = [i for i in range(0, len(label_vals_flattened_complete_trajectories))
                           if label_vals_flattened_complete_trajectories[i] == label]

                if len(indices) < MIN_TRAM_LENGTH:
                    tram = float('nan')
                else:
                    normalized_data_for_label = normalized_data_array[indices,] # get the corresponding data
                    images = [image_vals_flattened_complete_trajectories[i] for i in indices]
                    normalized_data_for_label = normalized_data_for_label[np.argsort(images),] # order by time
                    normalized_data_dict = {data_keys[i] : normalized_data_for_label[:,i] for i in range(0,len(data_keys)) }
                    tram = self.compute_TrAM(normalized_data_dict, euclidian_pairs)

            tram_dict.update({label : tram})

        # place the measurements in the workspace for each image
        for img, labels in zip(img_numbers, label_vals):
            workspace.measurements.add_measurement(obj_name, FULL_TRAM_MEAS_NAME,
                                                   [tram_dict[label] for label in labels], image_set_number=img)

        # store the non-nan TrAM values for the histogram display
        workspace.display_data.tram_values = [value for value in tram_dict.values() if not np.isnan(value)]

        pass

    def compute_TrAM(self, normalized_values_dict, euclidian):
        """
        
        :param normalized_values_dict: keys are feature names, values are normalized feature values across the track 
        :param euclidian list of pairs (tuples) of XY features which should be treated as Euclidian in the computation
        :return: TrAM value for this trajectory
        """
        def compute_single_aberration(normalized_values):
            """
            Figure out the deviation from smooth at each time point
            :param normalized_values: time series of values, normalized to the typical deviation
            :return: list of absolute deviation values from smooth
            """
            import scipy.interpolate as interp

            n = len(normalized_values)
            xs = np.array(range(1, n+1), float)
            num_knots = self.num_knots.get_value()
            knot_deltas = (n-1.0)/(num_knots+1.0)
            knot_locs = 1 + np.array(range(1, num_knots)) * knot_deltas

            try:
                interp_func = interp.LSQUnivariateSpline(xs, normalized_values, knot_locs)
                smoothed_vals = interp_func(xs)
            except ValueError:
                smoothed_vals = np.zeros(len(xs)) + float('nan') # return nan array

            return abs(normalized_values - smoothed_vals)

        # compute aberrations for each of the features
        aberration_dict = {feat_name : compute_single_aberration(np.array(values))
                           for feat_name, values in normalized_values_dict.items()}

        # now combine them with the appropriate power
        aberration_array = np.column_stack(aberration_dict.values())

        p = self.p.get_value()

        # handle Euclidian weightings
        num_euclidian = len(euclidian)
        if num_euclidian != 0:
            column_names = aberration_dict.keys()
            remaining_features = list(column_names)

            column_list = list() # we will accumulate data here
            weight_list = list() # will accumulate weights here

            for x, y in euclidian:
                # find data columns
                x_col = next(i for i, val in enumerate(column_names) if x == val)
                y_col = next(i for i, val in enumerate(column_names) if y == val)

                euclidian_vec = np.sqrt(np.apply_along_axis(np.mean, 1, aberration_array[:,(x_col,y_col)]))
                column_list.append(euclidian_vec)
                weight_list.append(2) # 2 data elements used to weight is twice the usual

                # remove the column names from remaining features
                remaining_features.remove(x)
                remaining_features.remove(y)

            # all remaining features have weight 1
            for feature_name in remaining_features:
                col = next(i for i, val in enumerate(column_names) if val == feature_name)
                column_list.append(aberration_array[:,col])
                weight_list.append(1)

            data_array = np.column_stack(column_list) # make array
            weight_array = np.array(weight_list, float)
            weight_array = weight_array / np.sum(weight_array) # normalize weights
            weight_matrix = np.diag(weight_array)

            pwr = np.power(data_array, p)
            weighted_means = np.apply_along_axis(np.sum, 1, np.matmul(pwr, weight_matrix))
            tram = np.max(np.power(weighted_means, 1.0/p))
        else:
            pwr = np.power(aberration_array, p)
            means = np.apply_along_axis(np.mean, 1, pwr)
            tram = np.max(np.power(means, 1.0/p))

        return tram

    @staticmethod
    def compute_typical_deviations(values_dict, labels_vec, image_vec):
        """
        Compute the median absolute temporal difference in each of the features across all tracks
        
        :param values_dict: keys are feature names, values are lists of data values across images and tracks
        :param labels_vec: A list of track labels corresponding to data values in their arrays
        :param image_vec: A list of image numbers corresponding to data values in their arrays
        :return: dictionary whose keys are feature names and values are median absolute differences
        """
        # input is a list of time series lists
        def compute_median_abs_deviation(values_lists):
            return np.median(abs(np.diff(reduce(np.append, values_lists))))

        # mapping from label to indices
        labels_dict = dict()
        labels_set = set(labels_vec)
        for label in labels_set:
            indices = np.flatnonzero(labels_vec == label) # which
            labels_dict.update({label : indices})

        result = dict()
        # for each feature get the deltas in time
        for feat_name, values in values_dict.items():
            all_diffs = list()
            for label, indices in labels_dict.items():
                data = [values[i] for i in indices]
                images = [image_vec[i] for i in indices]
                z = sorted(zip(images, data)) # get them in time order
                ordered_data = [data for _, data in z]
                all_diffs.append(ordered_data)
            mad = compute_median_abs_deviation(all_diffs)
            result.update({feat_name : mad})


        return result

    @classmethod
    def Determine_Euclidian_pairs(cls, features):
        """
        Look for any pairs that end in "_X" and "_Y" or have "_X_" and "_Y_" within them
        :param features:list of names 
        :return: list of tubples containing pairs of names which are Euclidian
        """

        # first find all the ones with a "_X$"
        features_X_1 = [feature for feature in features if re.search("_X$", feature)]
        features_X_2 = [feature for feature in features if re.search("_X_", feature)]

        # get corresponding pairs
        paired_1 = [(feature, re.sub("_X$", "_Y", feature)) for feature in features_X_1]
        paired_2 = [(feature, re.sub("_X_", "_Y_", feature)) for feature in features_X_2]

        pairs = paired_1 + paired_2

        # only return pairs where the Y feature exists
        return [(x, y) for x, y in pairs if y in features]

    # Get the selected measurements, restricted to those which start with the object name
    def get_selected_tram_measurements(self):
        # get what was selected by the user
        selections = self.tram_measurements.get_selections()

        # get the object set to work on
        object_name = self.object_name.get_value()

        return [sel for sel in selections if sel.startswith(object_name)]

    def get_measurement_columns(self, pipeline):
        return [(self.object_name.get_value(), FULL_TRAM_MEAS_NAME, cpmeas.COLTYPE_FLOAT)]

    def get_categories(self, pipeline, object_name):
        if object_name == self.object_name.get_value():
            return [CAT_TRAM]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.object_name.get_value() and category == CAT_TRAM:
            return [MEAS_TRAM]
        return []

    def get_measurement_scales(self, pipeline, object_name, category, feature, image_name):
        return [self.wants_XY_Euclidian, self.p, self.num_knots]


    def is_aggregation_module(self): # todo - not sure what to return here
        """If true, the module uses data from other imagesets in a group

        Aggregation modules perform operations that require access to
        all image sets in a group, generally resulting in an aggregation
        operation during the last image set or in post_group. Examples are
        TrackObjects, MakeProjection and CorrectIllumination_Calculate.
        """
        return True