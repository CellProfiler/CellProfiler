import itertools
import logging
import re
from collections import Counter, defaultdict

import numpy

import cellprofiler.measurement as cpmeas
import cellprofiler.module as cpm
import cellprofiler.setting as cps
from cellprofiler.measurement import M_NUMBER_OBJECT_NUMBER
from cellprofiler.modules import trackobjects
from cellprofiler.setting import MeasurementMultiChoiceForCategory
# todo: make sure coherent error message is produced when num timepoints is < MIN_TRAM_LENGTH
# todo: see if you can access the class attribute MIN_TRAM_LENGTH instead of 6 for the below __doc__ string

__doc__ = """
<b>TrackQuality</b> provides tracking quality metrics. TrAM (Tracking
Aberration Measure) is based on temporal smoothness of features measured
across each object's trajectory.
<hr>
This module must be placed downstream of a module that identifies objects
(e.g., <b>IdentifyPrimaryObjects</b>) and a <b>TrackObjects</b> that tracks
them. There must be at least 6 frames to perform a TrAM analysis. The TrAM
statistic reflects how typical the maximum deviation from smooth time series
a chosen set of measurements are. Typical fluctuations are determined from
measurement differences in adjacent time points of objects whose trajectories
are complete and without splitting events.

<p><b>TODO</b>For an example pipeline using TrAM see the CellProfiler
<a href="http://www.cellprofiler.org/examples.html#TrAM">Examples</a> webpage.</p>

<h4>Available measurements</h4>
<ul>
<li><i>TrAM:</i> The TrAM value for the trajectory. Values near 1 are typical
for a good trajectory. Large values (typically 3 or higher) are more likely
to correspond to aberrant tracks. The value <i>None</i> is assigned to objects
with partial tracks or those for whom <i>Is_Parent</i> is 1. A histogram of
all computed TrAM values is displayed to help define a cutoff.</li>
<li><i>Labels:</i> Each tracked item that has a lineage from the first to the
last frame is assigned a TrAM label on the last frame. If the final object
does not arise from a split during tracking, then it has this same unique label
for its entire track. That label is not assigned to any other objects. But if
an object arises from a split, then its ancestor object(s) will be assigned
multiple labels (combined from its progeny). These labels are separated by
a "|" symbol.</li>
<li><i>Is_Parent:</i> If the object splits into daughters during its track then 
flag will be 1, and <i>Labels</i> will be a list of two or more labels.
Otherwise it is 0 and <i>Labels</i> is a list containing one label.</li>
<li><i>Split_Trajectory:</i> If the object arose from an ancestor whose trajectory
split, then this value is 1. Otherwise it is 0.</li>
</ul>
<p>
See Patsch, K <i>et al.</i>, <a href=https://www.nature.com/articles/srep34785>Single cell
dynamic phenotyping</a>, Scientific Reports 6:34785 (2016)
"""

logger = logging.getLogger(__name__)

class MeasureTrackQuality(cpm.Module):
    module_name = "MeasureTrackQuality"
    category = "Measurement"
    variable_revision_number = 1

    CAT_MEASURE_TRACK_QUALITY = "MeasureTrackQuality"
    MEAS_TRAM = "TrAM"
    MEAS_LABELS = "Labels"
    MEAS_PARENT = "Is_Parent"
    MEAS_SPLIT = "Split_Trajectory"
    FULL_TRAM_MEAS_NAME = "{}_{}".format(CAT_MEASURE_TRACK_QUALITY, MEAS_TRAM)
    FULL_LABELS_MEAS_NAME = "{}_{}".format(CAT_MEASURE_TRACK_QUALITY, MEAS_LABELS)
    FULL_PARENT_MEAS_NAME = "{}_{}".format(CAT_MEASURE_TRACK_QUALITY, MEAS_PARENT)
    FULL_SPLIT_MEAS_NAME = "{}_{}".format(CAT_MEASURE_TRACK_QUALITY, MEAS_SPLIT)
    IMAGE_NUM_KEY = "Image"
    MIN_TRAM_LENGTH = 6 # minimum number of timepoints to calculate TrAM
    MIN_NUM_KNOTS = 3

    LABELS_KEY = "labels"
    IMAGE_NUMS_KEY = "image_nums"
    OBJECT_NUMS_KEY = "object_nums"
    PARENT_OBJECT_NUMS_KEY = "parent_object_nums"
    TRAM_KEY = "TrAM"
    SPLIT_KEY = "split"
    PARENT_KEY = "parent"

    def create_settings(self):
        # for them to choose the tracked objects
        # todo: do not allow them to select if there are not 6 or more time points. Put this in description.
        self.object_name = cps.ObjectNameSubscriber(
                "Tracked objects", cps.NONE, doc="""
            Select the tracked objects for computing TrAM.""")

        # which measurements will go into the TrAM computation
        self.tram_measurements = MeasurementMultiChoiceForCategory(
            "TrAM measurements", category_chooser=self.object_name, doc="""
            These are measurements for the selected tracked objects which
            will be used in the TrAM computation. At least one must be selected.""")

        # Treat X-Y value pairs as isotropic in the TrAM measure?
        self.isotropic = cps.Binary(
            'Isotropic XY metric?', True, doc="""
            If selected (the default) then measurements that are available
            as X-Y pairs (e.g. location) will be have an isotropic
            metric applied in TrAM. Note that the X-Y-Z extension of this feature
            is not currently available.
            """)

        # number of spline knots
        self.num_knots = cps.Integer(
            "Number of spline knots", 4, minval=self.MIN_NUM_KNOTS, doc="""
            The number of knots (indpendent values) used
            when computing smoothing splines. This should be around 1/5th the number
            of frames for reasonably oversampled time lapse sequences, and must be 3
            or greater. It is approximately the maximum number of wiggles expected in
            well-tracked trajectories
            """)

        # TrAM exponent
        self.tram_exponent = cps.Float(
            "TrAM exponent", 0.5, minval=0.01, maxval=1.0, doc="""
            This number is between 0.01 and 1 (default 0.5), and specifies how
            strongly simultaneous sudden changes in multiple features synergize in
            the TrAM metric. A lower value signifies higher synergy (at the risk of
            missing tracking failures that are reflected in only some of the features).
            """)

    def settings(self):
        return [self.object_name, self.tram_measurements, self.isotropic, self.num_knots, self.tram_exponent]

    def validate_module(self, pipeline):
        '''Make sure that the user has selected at least one measurement for TrAM and that there are tracking data.'''
        if len(self.get_selected_tram_measurements()) == 0:
            raise cps.ValidationError(
                    "Please select at least one TrAM measurement for tracking of {}".format(self.object_name.value),
                    self.tram_measurements)

        # check on available tracking columns for the selected object
        obj_name = self.object_name.value
        mc = pipeline.get_measurement_columns()
        num_tracking_cols = len([entry for entry in mc if entry[0] == obj_name and entry[1].startswith(trackobjects.F_PREFIX)])
        if num_tracking_cols == 0:
            msg = "No {} data available for {}. Please select an object with tracking data.".format(trackobjects.F_PREFIX, obj_name)
            raise cps.ValidationError(msg, self.object_name)

    def run(self, workspace):
        pass

    def display_post_group(self, workspace, figure):
        if self.show_window:
            figure.set_subplots((1,1))
            figure.subplot_histogram(0, 0, workspace.display_data.tram_values, bins=40, xlabel="TrAM",
                                     title="TrAM for {}".format(self.object_name.value))

    def post_group(self, workspace, grouping):
        self.show_window = True

        measurements = workspace.measurements
        obj_name = self.object_name.value # the object the user has selected

        # get the image numbers
        group_number = grouping["Group_Number"]
        groupings = workspace.measurements.get_groupings(grouping)
        img_numbers = sum([numbers for group, numbers in groupings if int(group["Group_Number"]) == group_number], [])

        num_images = len(img_numbers)

        # get vector of tracking label for each data point
        feature_names = measurements.get_feature_names(obj_name)
        tracking_label_feature_name = [name for name in feature_names
                                       if name.startswith("{}_{}".format(trackobjects.F_PREFIX, trackobjects.F_LABEL))][0]
        label_vals = measurements.get_measurement(obj_name, tracking_label_feature_name, img_numbers)
        label_vals_flattened_all = numpy.concatenate(label_vals).ravel().tolist()
        # determine which indexes we should keep. Get rid of any nan label values
        not_nan_indices = [i for i, label in enumerate(label_vals_flattened_all) if not numpy.isnan(label)]
        label_vals_flattened = [label_vals_flattened_all[i] for i in not_nan_indices] # excludes nan

        # convenience function to flatten and remove values corresponding to nan labels
        def extract_flattened_measurements_for_valid_labels(lol):
            return [numpy.concatenate(lol).tolist()[i] for i in not_nan_indices]

        # function to get a tuple dictionary entry relating feature name with data values
        def get_feature_values_tuple(sel):
            feat_obj_name, feat_name = sel.split("|")
            vals = measurements.get_measurement(feat_obj_name, feat_name, measurements.get_image_numbers())
            vals_flattened = extract_flattened_measurements_for_valid_labels(vals)
            return (feat_name, vals_flattened)

        # get all the data for TrAM
        selections = self.get_selected_tram_measurements() # measurements that the user wants to run TrAM on
        all_values_dict = dict(get_feature_values_tuple(sel) for sel in selections)
        # determine if there are any potential isotropic (XY) pairs
        if self.isotropic.value:
            isotropic_pairs = MeasureTrackQuality.Determine_Isotropic_pairs(all_values_dict.keys())
        else:
            isotropic_pairs = []

        # sanity check: make sure all vectors have the same length
        vec_lengths = set([len(value) for value in all_values_dict.values()])
        assert len(vec_lengths) == 1, "Measurement vectors have differing lengths"

        # get vector of image numbers into the dict
        counts = [len([v for v in x if not numpy.isnan(v)]) for x in label_vals] # number of non-nan labels at each time point
        image_vals = [[image for _ in range(count)] for image, count in zip(img_numbers, counts)] # repeat image number
        image_vals_flattened = sum(image_vals, [])

        # determine max lifetime by label so we can select different object behaviors
        lifetime_feature_name = [name for name in feature_names
                                 if name.startswith("{}_{}".format(trackobjects.F_PREFIX, trackobjects.F_LIFETIME))][0]
        lifetime_vals_flattened =\
            extract_flattened_measurements_for_valid_labels(measurements.get_measurement(obj_name,
                                                                                         lifetime_feature_name,
                                                                                         img_numbers))
        max_lifetime_by_label = dict(max(lifetimes)
                                     for label, lifetimes
                                     in itertools.groupby(zip(label_vals_flattened, lifetime_vals_flattened),
                                                          lambda x: x[0]))


        # Labels for objects that are tracked the whole time.
        label_counts = Counter(label_vals_flattened) # dict with count of each label
        labels_for_complete_trajectories = [label for label in max_lifetime_by_label.keys()
                                            if max_lifetime_by_label[label] == num_images
                                            and label_counts[label] == num_images]
        # labels for objects there the whole time but result from splitting
        labels_for_split_trajectories = [label for label in max_lifetime_by_label.keys()
                                         if max_lifetime_by_label[label] == num_images
                                         and label_counts[label] > num_images
                                         and not numpy.isnan(label)]


        # create dictionary to translate from label to object number in last frame. This is how we will store results.
        object_nums = measurements.get_measurement(obj_name, M_NUMBER_OBJECT_NUMBER, img_numbers) # list of lists
        object_nums_flattened = extract_flattened_measurements_for_valid_labels(object_nums)
        object_count_by_image = {img_num:len(v) for img_num, v in zip(img_numbers, object_nums)}

        # create a mapping from object number in an image to its index in the data array for later
        index_by_img_and_object = {(img_num, obj_num): index for img_num, obj_nums in zip(img_numbers, object_nums)
                                   for index, obj_num in enumerate(obj_nums)}

        last_image_num = img_numbers[-1]
        # todo: why is the below variable no longer needed?
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
        tad = MeasureTrackQuality.compute_typical_deviations(all_values_dict_complete_trajectories,
                                                               label_vals_flattened_complete_trajectories,
                                                               image_vals_flattened_complete_trajectories)


        # put all the data into a 2D array and normalize by typical deviations
        all_data_array = numpy.column_stack(all_values_dict.values())
        tram_feature_names = all_values_dict_complete_trajectories.keys()
        inv_devs = numpy.diag([1 / tad[k] for k in tram_feature_names]) # diagonal matrix of inverse typical deviation
        normalized_all_data_array = numpy.dot(all_data_array, inv_devs) # perform the multiplication

        # this is how we identify our TrAM measurements to objects
        next_available_tram_label = 0

        # compute TrAM for each complete trajectory. Store result by object number in last frame
        tram_dict = dict()
        for label in labels_for_complete_trajectories:
            indices = [i for i, lab in enumerate(label_vals_flattened) if lab == label]

            if len(indices) < self.MIN_TRAM_LENGTH: # not enough data points
                tram = None
            else:
                tram = self.compute_TrAM(tram_feature_names, normalized_all_data_array,
                                         image_vals_flattened, indices, isotropic_pairs)

            obj_nums = {image_vals_flattened[i] : object_nums_flattened[i] for i in indices} # pairs of image and object
            tram_dict.update({next_available_tram_label : {self.TRAM_KEY : tram, self.OBJECT_NUMS_KEY : obj_nums, self.SPLIT_KEY : 0}})
            next_available_tram_label += 1


        # now compute TrAM for split trajectories
        tracking_info_dict = dict()
        tracking_info_dict[self.LABELS_KEY] = label_vals_flattened
        tracking_info_dict[self.IMAGE_NUMS_KEY] = image_vals_flattened
        tracking_info_dict[self.OBJECT_NUMS_KEY] = object_nums_flattened

        parent_object_text_start = "{}_{}".format(trackobjects.F_PREFIX, trackobjects.F_PARENT_OBJECT_NUMBER)
        parent_object_feature = next(feature_name for feature_name in feature_names
                                     if feature_name.startswith(parent_object_text_start))
        tracking_info_dict[self.PARENT_OBJECT_NUMS_KEY] = \
            extract_flattened_measurements_for_valid_labels(measurements.get_measurement(obj_name,
                                                                                         parent_object_feature,
                                                                                         img_numbers))

        split_trajectories_tram_dict = \
            self.evaluate_tram_for_split_objects(labels_for_split_trajectories, tram_feature_names,
                                                 isotropic_pairs, normalized_all_data_array,
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
            tram = traj_dict[self.TRAM_KEY]
            split_flag = traj_dict[self.SPLIT_KEY]
            for img_num, object_num in traj_dict[self.OBJECT_NUMS_KEY].iteritems(): # every object across images for this tram
                index = index_by_img_and_object[(img_num, object_num)]
                result_dict = results_to_store_by_img[img_num][index]

                if result_dict is None:
                    result_dict = dict() # initialize
                    results_to_store_by_img[img_num][index] = result_dict # store it
                    result_dict.update({self.PARENT_KEY:0})
                    result_dict.update({self.TRAM_KEY:tram})
                    result_dict.update({self.LABELS_KEY:[tram_label]})
                else: # if there is already a TRAM_KEY then we are a parent and don't have a valid TrAM
                    result_dict.update({self.PARENT_KEY:1})
                    result_dict.update({self.TRAM_KEY:None})
                    previous_list = result_dict[self.LABELS_KEY]
                    previous_list.append(tram_label)

                result_dict.update({self.SPLIT_KEY: split_flag})

        # Loop over all images and save out
        tram_values_to_save = list()
        parent_values_to_save = list()
        split_values_to_save = list()
        label_values_to_save = list()

        for img_num, vec in results_to_store_by_img.iteritems():
            tram_values_to_save.append([get_element_or_default_for_None(v, self.TRAM_KEY, None) for v in vec])
            parent_values_to_save.append([get_element_or_default_for_None(v, self.PARENT_KEY, None) for v in vec])
            split_values_to_save.append([get_element_or_default_for_None(v, self.SPLIT_KEY, None) for v in vec])
            label_values_to_save.append([get_element_or_default_for_None(v, self.LABELS_KEY, None) for v in vec])

        img_nums = results_to_store_by_img.keys()
        workspace.measurements.add_measurement(obj_name, self.FULL_TRAM_MEAS_NAME, tram_values_to_save, image_set_number=img_nums)
        workspace.measurements.add_measurement(obj_name, self.FULL_PARENT_MEAS_NAME, parent_values_to_save, image_set_number=img_nums)
        workspace.measurements.add_measurement(obj_name, self.FULL_SPLIT_MEAS_NAME, split_values_to_save, image_set_number=img_nums)
        workspace.measurements.add_measurement(obj_name, self.FULL_LABELS_MEAS_NAME, label_values_to_save, image_set_number=img_nums)

        # store the existing TrAM values for the histogram display
        workspace.display_data.tram_values = [d.get(self.TRAM_KEY)
                                              for d in tram_dict.values() if d.get(self.TRAM_KEY) is not None]


    def compute_TrAM(self, tram_feature_names, normalized_data_array, image_vals_flattened, indices, isotropic_pairs):
        """
        Compute the TrAM statistic for a single trajectory
        
        :param tram_feature_names: Names of the features to use (in order of the columns in normalized_data_array) 
        :param normalized_data_array: Source of data (normalized to typical absolute deviations). Columns correspond
        to TrAM features, and rows are for all objects across images
        :param image_vals_flattened: The image numbers corresponding to rows in normalized_data_array
        :param indices: The rows in normalized_data_array which are relevant to this trajectory
        :param isotropic_pairs: List of pairs of features which should be treated with a Euclidian metric
        :return: The computed TrAM value
        """
        normalized_data_for_label = normalized_data_array[indices,:]  # get the corresponding data
        images = [image_vals_flattened[i] for i in indices]

        normalized_data_for_label = normalized_data_for_label[numpy.argsort(images),]  # order by image
        normalized_values_dict = {tram_feature_names[i]: normalized_data_for_label[:, i] for i in range(0, len(tram_feature_names))}

        def compute_single_aberration(normalized_values):
            """
            Figure out the deviation from smooth at each time point
            :param normalized_values: time series of values, normalized to the typical deviation
            :return: list of absolute deviation values from smooth
            """
            import scipy.interpolate as interp

            n = len(normalized_values)
            xs = numpy.array(range(1, n + 1), float)
            num_knots = self.num_knots.get_value()
            knot_deltas = (n-1.0)/(num_knots+1.0)
            knot_locs = 1 + numpy.array(range(1, num_knots)) * knot_deltas

            try:
                interp_func = interp.LSQUnivariateSpline(xs, normalized_values, knot_locs)
                smoothed_vals = interp_func(xs)
            except ValueError:
                smoothed_vals = numpy.zeros(len(xs)) + numpy.nan # return nan array

            return abs(normalized_values - smoothed_vals)

        # compute aberrations for each of the features
        aberration_dict = {feat_name : compute_single_aberration(numpy.array(values))
                           for feat_name, values in normalized_values_dict.items()}

        # now combine them with the appropriate power
        aberration_array = numpy.column_stack(aberration_dict.values())

        p = self.tram_exponent.get_value()

        # handle Euclidian weightings
        num_isotropic = len(isotropic_pairs)
        if num_isotropic != 0:
            column_names = aberration_dict.keys()
            remaining_features = list(column_names)

            column_list = list() # we will accumulate data here
            weight_list = list() # will accumulate weights here

            for x, y in isotropic_pairs:
                # find data columns
                x_col = next(i for i, val in enumerate(column_names) if x == val)
                y_col = next(i for i, val in enumerate(column_names) if y == val)

                isotropic_vec = numpy.sqrt(numpy.apply_along_axis(numpy.mean, 1, aberration_array[:, (x_col, y_col)]))
                column_list.append(isotropic_vec)
                weight_list.append(2) # 2 data elements used to weight is twice the usual

                # remove the column names from remaining features
                remaining_features.remove(x)
                remaining_features.remove(y)

            # all remaining features have weight 1
            for feature_name in remaining_features:
                col = next(i for i, val in enumerate(column_names) if val == feature_name)
                column_list.append(aberration_array[:,col])
                weight_list.append(1)

            data_array = numpy.column_stack(column_list) # make array
            weight_array = numpy.array(weight_list, float)
            weight_array = weight_array / numpy.sum(weight_array) # normalize weights
            weight_matrix = numpy.diag(weight_array)

            pwr = numpy.power(data_array, p)
            weighted_means = numpy.apply_along_axis(numpy.sum, 1, numpy.matmul(pwr, weight_matrix))
            tram = numpy.max(numpy.power(weighted_means, 1.0 / p))
        else:
            pwr = numpy.power(aberration_array, p)
            means = numpy.apply_along_axis(numpy.mean, 1, pwr)
            tram = numpy.max(numpy.power(means, 1.0 / p))

        return tram

    def evaluate_tram_for_split_objects(self, labels_for_split_trajectories, tram_feature_names, isotropic_pairs,
                                        normalized_data_array, tracking_info_dict, next_available_tram_label):
        """
        Compute TrAM results for objects that have split trajectories        
        :param labels_for_split_trajectories: TrackObjects labels for trajectories that split.
        :param tram_feature_names:  The feature names that are used to compute TrAM.
        :param isotropic_pairs: List of feature pairs (XY) to be Euclidianized.
        :param normalized_data_array: Data for the TrAM features, normalized by typical absolute deviation.
        :param tracking_info_dict: Dictionary of other relevant information about the objects.
        :param next_available_tram_label: Tram label number. We increment this as we use it.
        :return: Dictionary whose keys are TrAM labels and values are dictionaries containing values
        for the keys TRAM_KEY, OBJECT_NUMS_KEY, SPLIT_KEY
        """

        label_vals_flattened = tracking_info_dict[self.LABELS_KEY]
        image_vals_flattened = tracking_info_dict[self.IMAGE_NUMS_KEY]
        object_nums_flattened = tracking_info_dict[self.OBJECT_NUMS_KEY]
        parent_object_nums_flattened = tracking_info_dict[self.PARENT_OBJECT_NUMS_KEY]

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

        # Compute TrAM for each label of split objects. They will all have
        # a complete set of predecessor objects going from the end to the start since
        # they were filtered to have a max lifetime equal to the number of frames.
        # Here we piece together the entire trajectory for each object and compute TrAM.
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
                                         indices_list, isotropic_pairs)

                # for each image number, the corresponding object number
                obj_nums = dict(zip([image_vals_flattened[i] for i in indices_list], object_nums_list))

                result.update({next_available_tram_label: {self.TRAM_KEY:tram, self.OBJECT_NUMS_KEY:obj_nums,
                                                           self.SPLIT_KEY:1}})
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
            return numpy.median(abs(numpy.diff(reduce(numpy.append, values_lists))))

        # mapping from label to indices
        labels_dict = dict()
        labels_set = set(labels_vec)
        for label in labels_set:
            indices = [i for i, lab in enumerate(labels_vec) if lab == label] # which match
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

    @staticmethod
    def Determine_Isotropic_pairs(features):
        """
        Look for any pairs that end in "_X" and "_Y" or have "_X_" and "_Y_" within them
        :param features:list of names 
        :return: list of tubples containing pairs of names which can be paired using an isotropic (Euclidian) metric
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
        object_name = self.object_name.value

        return [sel for sel in selections if sel.startswith(object_name)]

    def get_measurement_columns(self, pipeline):
        return [(self.object_name.value, self.FULL_TRAM_MEAS_NAME, cpmeas.COLTYPE_FLOAT),
                (self.object_name.value, self.FULL_PARENT_MEAS_NAME, cpmeas.COLTYPE_FLOAT),
                (self.object_name.value, self.FULL_SPLIT_MEAS_NAME, cpmeas.COLTYPE_FLOAT),
                (self.object_name.value, self.FULL_LABELS_MEAS_NAME, cpmeas.COLTYPE_BLOB)]

    def get_categories(self, pipeline, object_name):
        if object_name == self.object_name.value:
            return [self.CAT_MEASURE_TRACK_QUALITY]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.object_name.value and category == self.CAT_MEASURE_TRACK_QUALITY:
            return [self.MEAS_TRAM, self.MEAS_PARENT, self.MEAS_SPLIT, self.MEAS_LABELS]
        return []

    def is_aggregation_module(self): # todo - not sure what to return here
        """If true, the module uses data from other imagesets in a group

        Aggregation modules perform operations that require access to
        all image sets in a group, generally resulting in an aggregation
        operation during the last image set or in post_group. Examples are
        TrackObjects, MakeProjection and CorrectIllumination_Calculate.
        """
        return True

