from cellprofiler.modules import trackobjects
import numpy as np
import itertools as it
import cellprofiler.module as cpm
import cellprofiler.setting as cps
import cellprofiler.measurement as cpmeas
from collections import Counter, defaultdict
import re
from cellprofiler.measurement import M_NUMBER_OBJECT_NUMBER
import logging

CAT_TRAM = "TrAM"
MEAS_TRAM = "TrAM"
MEAS_LABELS = "Labels"
MEAS_PARENT = "Is_Parent"
MEAS_SPLIT = "Split_Trajectory"
FULL_TRAM_MEAS_NAME = "%s_%s" % (CAT_TRAM, MEAS_TRAM)
FULL_LABELS_MEAS_NAME = "%s_%s" % (CAT_TRAM, MEAS_LABELS)
FULL_PARENT_MEAS_NAME = "%s_%s" % (CAT_TRAM, MEAS_PARENT)
FULL_SPLIT_MEAS_NAME = "%s_%s" % (CAT_TRAM, MEAS_SPLIT)
IMAGE_NUM_KEY = "Image"
MIN_TRAM_LENGTH = 6 # minimum number of timepoints to calculate TrAM

LABELS_KEY = "labels"
IMAGE_NUMS_KEY = "image_nums"
OBJECT_NUMS_KEY = "object_nums"
PARENT_OBJECT_NUMS_KEY = "parent_object_nums"
TRAM_KEY = "TrAM"
SPLIT_KEY = "split"
PARENT_KEY = "parent"

# todo
__doc__ = ""

logger = logging.getLogger(__name__)

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
                    "Please select at least one TrAM measurement for tracking of %s" % self.object_name.get_value(),
                    self.tram_measurements)
        # todo: check for tracking data for the chosen object

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
        if self.show_window:
            figure.set_subplots((1,1))
            figure.subplot_histogram(0, 0, workspace.display_data.tram_values, bins=40, xlabel="TrAM",
                                     title="TrAM for %s" % self.object_name.get_value())



    def post_group(self, workspace, grouping):
        self.show_window = True

        measurements = workspace.measurements
        obj_name = self.object_name.get_value() # the object the user has selected
        img_numbers = measurements.get_image_numbers()
        num_images = len(img_numbers)

        def flatten_list_of_lists(lol): # todo move outside
            return list(it.chain.from_iterable(lol))

        # get vector of tracking label for each data point
        feature_names = measurements.get_feature_names(obj_name)
        tracking_label_feature_name = [name for name in feature_names
                                       if name.startswith("%s_%s" % (trackobjects.F_PREFIX, trackobjects.F_LABEL))][0]
        label_vals = measurements.get_measurement(obj_name, tracking_label_feature_name, img_numbers)
        label_vals_flattened_all = flatten_list_of_lists(label_vals)
        # determine which indexes we should keep. Get rid of any nan label values
        not_nan_indices = [i for i, label in enumerate(label_vals_flattened_all) if not np.isnan(label)]
        label_vals_flattened = [label_vals_flattened_all[i] for i in not_nan_indices] # excludes nan

        # convenience function to flatten and remove values corresponding to nan labels
        def extract_flattened_measurements_for_valid_labels(lol):
            return [flatten_list_of_lists(lol)[i] for i in not_nan_indices]

        # function to get a tuple dictionary entry relating feature name with data values
        def get_feature_values_tuple(sel):
            feat_obj_name, feat_name = sel.split("|")
            vals = measurements.get_measurement(feat_obj_name, feat_name, measurements.get_image_numbers())
            vals_flattened = extract_flattened_measurements_for_valid_labels(vals)
            return (feat_name, vals_flattened)

        # get all the data for TrAM
        selections = self.get_selected_tram_measurements() # measurements that the user wants to run TrAM on
        all_values_dict = dict(get_feature_values_tuple(sel) for sel in selections)
        # determine if there are any euclidian (XY) pairs
        if self.wants_XY_Euclidian.value:
            euclidian_pairs = TrAM.Determine_Euclidian_pairs(all_values_dict.keys())
        else:
            euclidian_pairs = []

        # sanity check: make sure all vectors have the same length
        vec_lengths = set([len(value) for value in all_values_dict.values()])
        assert len(vec_lengths) == 1

        # get vector of image numbers into the dict
        counts = [len([v for v in x if not np.isnan(v)]) for x in label_vals] # number of non-nan labels at each time point
        image_vals = [[image for _ in xrange(count)] for image, count in zip(img_numbers, counts)] # repeat image number
        image_vals_flattened = flatten_list_of_lists(image_vals)

        # determine max lifetime by label so we can select different cell behaviors
        lifetime_feature_name = [name for name in feature_names
                                 if name.startswith("%s_%s" % (trackobjects.F_PREFIX, trackobjects.F_LIFETIME))][0]
        lifetime_vals_flattened =\
            extract_flattened_measurements_for_valid_labels(measurements.get_measurement(obj_name,
                                                                                         lifetime_feature_name,
                                                                                         img_numbers))
        max_lifetime_by_label = dict(max(lifetimes)
                                     for label, lifetimes
                                     in it.groupby(zip(label_vals_flattened, lifetime_vals_flattened),
                                                   lambda x: x[0]))


        # These are the cells that are tracked the whole time.
        label_counts = Counter(label_vals_flattened) # dict with count of each label
        labels_for_complete_trajectories = [label for label in max_lifetime_by_label.keys()
                                            if max_lifetime_by_label[label] == num_images
                                            and label_counts[label] == num_images]
        # labels for cells there the whole time but result from splitting
        labels_for_split_trajectories = [label for label in max_lifetime_by_label.keys()
                                           if max_lifetime_by_label[label] == num_images
                                           and label_counts[label] > num_images
                                           and not np.isnan(label)]


        # create dictionary to translate from label to object number in last frame. This is how we will store results.
        object_nums = measurements.get_measurement(obj_name, M_NUMBER_OBJECT_NUMBER, img_numbers) # list of lists
        object_nums_flattened = extract_flattened_measurements_for_valid_labels(object_nums)
        object_count_by_image = {img_num:len(v) for img_num, v in zip(img_numbers, object_nums)}

        # create a mapping from object number in an image to its index in the data array for later
        index_by_img_and_object = {(img_num, obj_num): index for img_num, obj_nums in zip(img_numbers, object_nums)
                                   for index, obj_num in enumerate(obj_nums)}

        last_image_num = img_numbers[-1]
        last_frame_label_to_object_num =\
            {object_num : label for object_num, label, image_num in zip(object_nums_flattened, label_vals_flattened,
                                                                        image_vals_flattened)
             if image_num == last_image_num}

        # now restrict vectors only to labels of complete trajectories
        complete_trajectory_indices = [i for i, label in enumerate(label_vals_flattened) if label in labels_for_complete_trajectories]
        all_values_dict_complete_trajectories = {k : [v[i] for i in complete_trajectory_indices] for k, v in all_values_dict.items()}

        # compute typical inter-timepoint variation for complete trajectories only.
        label_vals_flattened_complete_trajectories = [label_vals_flattened[i] for i in complete_trajectory_indices]
        image_vals_flattened_complete_trajectories = [image_vals_flattened[i] for i in complete_trajectory_indices]
        tad = TrAM.compute_typical_deviations(all_values_dict_complete_trajectories,
                                              label_vals_flattened_complete_trajectories,
                                              image_vals_flattened_complete_trajectories)


        # put all the data into a 2D array and normalize by typical deviations
        all_data_array = np.column_stack(all_values_dict.values())
        tram_feature_names = all_values_dict_complete_trajectories.keys()
        inv_devs = np.diag([1/tad[k] for k in tram_feature_names]) # diagonal matrix of inverse typical deviation
        normalized_all_data_array = np.dot(all_data_array, inv_devs) # perform the multiplication

        # this is how we identify our TrAM measurements to cells
        next_available_tram_label = 0

        # todo: add flag for split vs complete trajectory

        # compute TrAM for each complete trajectory. Store result by object number in last frame
        tram_dict = dict()
        for label in labels_for_complete_trajectories:
            indices = [i for i, lab in enumerate(label_vals_flattened) if lab == label]

            if len(indices) < MIN_TRAM_LENGTH: # not enough data points
                tram = float('nan')
            else:
                tram = self.compute_TrAM(tram_feature_names, normalized_all_data_array,
                                         image_vals_flattened, indices, euclidian_pairs)

            obj_nums = {image_vals_flattened[i] : object_nums_flattened[i] for i in indices} # pairs of image and object
            tram_dict.update({next_available_tram_label : {TRAM_KEY : tram, OBJECT_NUMS_KEY : obj_nums, SPLIT_KEY : 0}})
            next_available_tram_label += 1


        # now compute TrAM for split trajectories
        tracking_info_dict = dict()
        tracking_info_dict[LABELS_KEY] = label_vals_flattened
        tracking_info_dict[IMAGE_NUMS_KEY] = image_vals_flattened
        tracking_info_dict[OBJECT_NUMS_KEY] = object_nums_flattened

        parent_object_text_start = "%s_%s" % (trackobjects.F_PREFIX, trackobjects.F_PARENT_OBJECT_NUMBER)
        parent_object_feature = next(feature_name for feature_name in feature_names
                                     if feature_name.startswith(parent_object_text_start))
        tracking_info_dict[PARENT_OBJECT_NUMS_KEY] = \
            extract_flattened_measurements_for_valid_labels(measurements.get_measurement(obj_name,
                                                                                         parent_object_feature,
                                                                                         img_numbers))

        split_trajectories_tram_dict = \
            self.get_full_track_data_for_split_cells(labels_for_split_trajectories, tram_feature_names,
                                                     euclidian_pairs, normalized_all_data_array,
                                                     tracking_info_dict, next_available_tram_label)
        tram_dict.update(split_trajectories_tram_dict) # store them with the others

        def get_element_or_default_for_None(x, index, default):
            if x is None:
                return default
            else:
                return x[index]

        results_to_store_by_img = {img_num: [None for _ in range(object_count_by_image[img_num])]
                                   for img_num in img_numbers} # Seems excessive. there must be a better way.

        # cycle through each tram computed
        for tram_label, traj_dict in tram_dict.iteritems():
            tram = traj_dict[TRAM_KEY]
            split_flag = traj_dict[SPLIT_KEY]
            for img_num, object_num in traj_dict[OBJECT_NUMS_KEY].iteritems(): # every object across images for this tram
                index = index_by_img_and_object[(img_num, object_num)]
                result_dict = results_to_store_by_img[img_num][index]

                if result_dict is None:
                    result_dict = dict() # initialize
                    results_to_store_by_img[img_num][index] = result_dict # store it
                    result_dict.update({PARENT_KEY: 0})
                    result_dict.update({TRAM_KEY: tram})
                    result_dict.update({LABELS_KEY: str(int(tram_label))})
                else: # if there is already a TRAM_KEY then we are a parent and don't have a valid TrAM
                    result_dict.update({PARENT_KEY: 1})
                    result_dict.update({TRAM_KEY: float('nan')})
                    result_dict.update({LABELS_KEY: "%s|%s" % (result_dict[LABELS_KEY], str(int(tram_label)))}) # append

                result_dict.update({SPLIT_KEY: split_flag})

        # Loop over all images and save out
        tram_values_to_save = list()
        parent_values_to_save = list()
        split_values_to_save = list()
        label_values_to_save = list()

        for img_num, vec in results_to_store_by_img.iteritems():
            tram_values_to_save.append([get_element_or_default_for_None(v, TRAM_KEY, float("nan")) for v in vec])
            parent_values_to_save.append([get_element_or_default_for_None(v, PARENT_KEY, float("nan")) for v in vec])
            split_values_to_save.append([get_element_or_default_for_None(v, SPLIT_KEY, float("nan")) for v in vec])
            label_values_to_save.append([get_element_or_default_for_None(v, LABELS_KEY, "") for v in vec])

        img_nums = results_to_store_by_img.keys()
        workspace.measurements.add_measurement(obj_name, FULL_TRAM_MEAS_NAME, tram_values_to_save, image_set_number=img_nums)
        workspace.measurements.add_measurement(obj_name, FULL_PARENT_MEAS_NAME, parent_values_to_save, image_set_number=img_nums)
        workspace.measurements.add_measurement(obj_name, FULL_SPLIT_MEAS_NAME, split_values_to_save, image_set_number=img_nums)
        workspace.measurements.add_measurement(obj_name, FULL_LABELS_MEAS_NAME, label_values_to_save, image_set_number=img_nums)

        # store the non-nan TrAM values for the histogram display
        workspace.display_data.tram_values = [d.get(TRAM_KEY) for d in tram_dict.values() if not np.isnan(d.get(TRAM_KEY))]


    def compute_TrAM(self, tram_feature_names, normalized_data_array, image_vals_flattened, indices, euclidian):
        # todo: update the below text
        """
        :param normalized_values_dict: keys are feature names, values are normalized feature values across the track 
        :param euclidian list of pairs (tuples) of XY features which should be treated as Euclidian in the computation
        :return: TrAM value for this trajectory
        """

        normalized_data_for_label = normalized_data_array[indices,:]  # get the corresponding data
        images = [image_vals_flattened[i] for i in indices]

        normalized_data_for_label = normalized_data_for_label[np.argsort(images),]  # order by image
        normalized_values_dict = {tram_feature_names[i]: normalized_data_for_label[:, i] for i in range(0, len(tram_feature_names))}

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

    def get_full_track_data_for_split_cells(self, labels_for_split_trajectories, tram_feature_names, euclidian,
                                            normalized_data_array, tracking_info_dict, next_available_tram_label):
        # todo: update text below
        """
        
        :param normalized_data_array: array whose columns correspond to TrAM features, in order of tram_feature_names
        :param tracking_info_dict: dictionary of tracking info (flattened arrays), e.g. object numbers, parents.
        :param labels_for_mitotic_trajectories: The tracking labels we should analyze
        :param next_available_tram_label: First available label number for tram (this gets updated by the method)
        :return: dictionary from object number in last frame to (TrAM value, TrAM label)
        """

        label_vals_flattened = tracking_info_dict[LABELS_KEY]
        image_vals_flattened = tracking_info_dict[IMAGE_NUMS_KEY]
        object_nums_flattened = tracking_info_dict[OBJECT_NUMS_KEY]
        parent_object_nums_flattened = tracking_info_dict[PARENT_OBJECT_NUMS_KEY]

        first_image_num = min(image_vals_flattened)
        last_image_num = max(image_vals_flattened)

        # Make a map from (image,object_number) to flattened array index so we can find parents
        img_obj_to_index = dict([((image_vals_flattened[i], object_nums_flattened[i]), i)
                                 for i in range(0, len(image_vals_flattened))])

        # Make a map from label to object number(s) for the last image. We will work backward from these
        object_nums_for_label_last_image = defaultdict(list) # need to store lists because there can be multiple
        # Restrict to labels for split trajectories and only last image
        for label, object_num, image_num in zip(label_vals_flattened, object_nums_flattened, image_vals_flattened):
            if image_num == last_image_num and label in labels_for_split_trajectories:
                object_nums_for_label_last_image[label].append(object_num)

        # Compute TrAM for each label in labels_for_split_cells. They will all have
        # a complete set of predecessor objects going from the end to the start since
        # they were filtered to have a max lifetime equal to the number of frames.
        # Here we piece together the entire trajectory for each cell and compute TrAM.
        # construct the object trajectory in terms of array indexes. These get placed
        # in an accumulator (list) that should be initialized as empty.
        def get_parent_indices(image_num, object_num, index_accum, object_num_accum):
            if image_num < first_image_num: return

            index = img_obj_to_index[(image_num, object_num)]
            parent_object_num = parent_object_nums_flattened[index]
            get_parent_indices(image_num - 1, parent_object_num, index_accum, object_num_accum) # recurse for all earlier

            index_accum.append(index)
            object_num_accum.append(object_num)

        # cycle through everything in our dict and compute tram. Store.
        result = dict()
        for label in object_nums_for_label_last_image.keys():
            for object_num_last_image in object_nums_for_label_last_image.get(label): # this is a list
                indices_list = list()
                object_nums_list = list()
                get_parent_indices(last_image_num, object_num_last_image, indices_list, object_nums_list)

                # Indices now contains the indices for the tracked object across images
                tram = self.compute_TrAM(tram_feature_names, normalized_data_array, image_vals_flattened,
                                         indices_list, euclidian)

                # for each image number, the corresponding object number
                obj_nums = dict(zip([image_vals_flattened[i] for i in indices_list], object_nums_list))

                result.update({next_available_tram_label: {TRAM_KEY: tram, OBJECT_NUMS_KEY: obj_nums, SPLIT_KEY: 1}})
                next_available_tram_label += 1

        return result

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
        return [(self.object_name.get_value(), FULL_TRAM_MEAS_NAME, cpmeas.COLTYPE_FLOAT),
                (self.object_name.get_value(), FULL_PARENT_MEAS_NAME, cpmeas.COLTYPE_FLOAT),
                (self.object_name.get_value(), FULL_SPLIT_MEAS_NAME, cpmeas.COLTYPE_FLOAT),
                (self.object_name.get_value(), FULL_LABELS_MEAS_NAME, cpmeas.COLTYPE_VARCHAR)]

    def get_categories(self, pipeline, object_name):
        if object_name == self.object_name.get_value():
            return [CAT_TRAM]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.object_name.get_value() and category == CAT_TRAM:
            return [MEAS_TRAM, MEAS_PARENT, MEAS_SPLIT, MEAS_LABELS]
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