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
MIN_NUM_KNOTS = 3

LABELS_KEY = "labels"
IMAGE_NUMS_KEY = "image_nums"
OBJECT_NUMS_KEY = "object_nums"
PARENT_OBJECT_NUMS_KEY = "parent_object_nums"
TRAM_KEY = "TrAM"
SPLIT_KEY = "split"
PARENT_KEY = "parent"

FLOAT_NAN = float('nan')

__doc__ = """
<b>TrAM</b> Provides a metric for tracking quality based on temporal
smoothness of features measured across the trajectory.
<hr>
This module must be placed downstream of a module that identifies objects
(e.g., <b>IdentifyPrimaryObjects</b>) and a <b>TrackObjects</b> that tracks
them. There must be at least %d frames to perform a TrAM analysis.

<p><b>TODO</b>For an example pipeline using TrAM see the CellProfiler
<a href="http://www.cellprofiler.org/examples.html#TrAM">Examples</a> webpage.</p>

<h4>Available measurements</h4>
<ul>
<li><i>TrAM:</i> The TrAM value for the trajectory. Values near 1 are typical
for a good trajectory. Large values (typically 3 or higher) are more likely
to correspond to aberrant tracks. The value <i>nan</i> is assigned to objects
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
flag will be 1, and <i>Labels</i> will contain two or more labels separated by "|".
Otherwise it is 0 and <i>Labels</i> has only one label.</li>
<li><i>Split_Trajectory:</i> If the object arose from an ancestor whose trajectory
split, then this value is 1. Otherwise it is 0.</li>
</ul>
The following inputs parameterize the TrAM computation:
<ul>
<li><i>Tracked objects:</i> Select the tracked objects on which TrAM will be computed</li>
<li><i>TrAM measurements</i> These are measurements for the selected tracked objects which
will be used in the TrAM computation. At least one must be selected. Note there may be
a delay of a few seconds between the selection of <i>Tracked objects</i> and the update of this
selection component.</li>
<li><i>Euclidian XY metric</i>If selected (the default) then measurements that are available
as X-Y pairs (e.g. location) will be have a Euclidian (isotropic) metric applied in TrAM.
Note that this feature is currently not available for X-Y-Z tracks.</li>
<li><i>Number of spline knots</i>The number of knots (indpendent values) used when computing
smoothing splines. This should be around 1/5th the number of frames for reasonably oversampled
time lapse sequences, and must be %d or greater. It is approximately the maximum number of
wiggles expected in well tracked trajectories.</li>
<li><i>TrAM exponent</i>This number is between 0 and 1 (default 0.5), and specifies how
strongly simultaneous sudden changes in multiple features synergize in the TrAM metric. A
lower value signifies higher synergy (at the risk of missing tracking failures that are
reflected in only some of the features).</li>
</ul>
<b>TODO: cite paper</b>
""" % (MIN_TRAM_LENGTH, MIN_NUM_KNOTS)

logger = logging.getLogger(__name__)

class TrAM(cpm.Module):
    module_name = "TrAM"
    category = "Object Processing"
    variable_revision_number = 1

    def create_settings(self):
        # for them to choose the tracked objects
        self.object_name = cps.ObjectNameSubscriber(
                "Tracked objects", cps.NONE, doc="""
            Select the tracked objects for computing TrAM.""")

        # which measurements will go into the TrAM computation
        self.tram_measurements = MeasurementMultiChoiceForCategory(
            "TrAM measurements", category_chooser=self.object_name, doc="""
            This setting defines the tracked quantities that will be used
            to compute the TrAM metric. At least one must be selected.""")

        self.wants_XY_Euclidian = cps.Binary(
            'Euclidian XY metric?', True, doc='''
            Euclidianize the metric for all measurements occurring in X-Y pairs
            ''')

        # spline knots
        self.num_knots = cps.Integer(
            "Number of spline knots", 4, minval=MIN_NUM_KNOTS, doc="""
            Number of knots to use in the spline fit to time series""")

        # TrAM exponent
        self.p = cps.Float(
            "TrAM exponent", 0.5, minval=0.01, maxval=1, doc="""
            The exponent used to combine different measurements.""")

    def settings(self):
        return [self.object_name, self.tram_measurements, self.wants_XY_Euclidian, self.num_knots, self.p]

    def validate_module(self, pipeline):
        '''Make sure that the user has selected at least one measurement for TrAM and that there are tracking data.'''
        if len(self.get_selected_tram_measurements()) == 0:
            raise cps.ValidationError(
                    "Please select at least one TrAM measurement for tracking of %s" % self.object_name.get_value(),
                    self.tram_measurements)

        # check on available tracking columns for the selected object
        obj_name = self.object_name.get_value()
        mc = pipeline.get_measurement_columns()
        num_tracking_cols = len([entry for entry in mc if entry[0] == obj_name and entry[1].startswith(trackobjects.F_PREFIX)])
        if num_tracking_cols == 0:
            msg = "No %s data available for %s. Please select an object with tracking data."\
                  % (trackobjects.F_PREFIX, obj_name)
            raise cps.ValidationError(msg, self.object_name)

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

        def flatten_list_of_lists(lol):
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
        image_vals = [[image for _ in range(count)] for image, count in zip(img_numbers, counts)] # repeat image number
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

        # compute TrAM for each complete trajectory. Store result by object number in last frame
        tram_dict = dict()
        for label in labels_for_complete_trajectories:
            indices = [i for i, lab in enumerate(label_vals_flattened) if lab == label]

            if len(indices) < MIN_TRAM_LENGTH: # not enough data points
                tram = FLOAT_NAN
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
            self.evaluate_tram_for_split_cells(labels_for_split_trajectories, tram_feature_names,
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
                    result_dict.update({TRAM_KEY: FLOAT_NAN})
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
        """
        Compute the TrAM statistic for a single trajectory
        
        :param tram_feature_names: Names of the features to use (in order of the columns in normalized_data_array) 
        :param normalized_data_array: Source of data (normalized to typical absolute deviations). Columns correspond
        to TrAM features, and rows are for all cells across images
        :param image_vals_flattened: The image numbers corresponding to rows in normalized_data_array
        :param indices: The rows in normalized_data_array which are relevant to this trajectory
        :param euclidian: List of pairs of features which should be treated with a Euclidian metric
        :return: The computed TrAM value
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
                smoothed_vals = np.zeros(len(xs)) + FLOAT_NAN # return nan array

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

    def evaluate_tram_for_split_cells(self, labels_for_split_trajectories, tram_feature_names, euclidian,
                                      normalized_data_array, tracking_info_dict, next_available_tram_label):
        """
        Compute TrAM results for cells that have split trajectories        
        :param labels_for_split_trajectories: TrackObjects labels for trajectories that split
        :param tram_feature_names:  The feature names that are used to compute TrAM
        :param euclidian: List of feature pairs (XY) to be Euclidianized
        :param normalized_data_array: Data for the TrAM features, normalized by typical absolute deviation
        :param tracking_info_dict: Dictionary of other relevant information about the cells
        :param next_available_tram_label: Tram label number. We increment this as we use it.
        :return: Dictionary whose keys are TrAM labels and values are dictionaries containing values
        for the keys TRAM_KEY, OBJECT_NUMS_KEY, SPLIT_KEY
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

    @staticmethod
    def Determine_Euclidian_pairs(features):
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

class MeasurementMultiChoiceForCategory(cps.MeasurementMultiChoice):
    '''A multi-choice setting for selecting multiple measurements within a given category'''

    def __init__(self, text, category_chooser, value='', *args, **kwargs):
        '''Initialize the measurement multi-choice

        At initialization, the choices are empty because the measurements
        can't be fetched here. It's done (bit of a hack) in test_valid.
        '''
        super(cps.MeasurementMultiChoice, self).__init__(text, [], value, *args, **kwargs)
        self.category_chooser = category_chooser

    def populate_choices(self, pipeline):
        #
        # Find our module
        #
        for module in pipeline.modules():
            for setting in module.visible_settings():
                if id(setting) == id(self):
                    break
        columns = pipeline.get_measurement_columns(module)

        def valid_mc(c):
            '''Disallow any measurement column with "," or "|" in its names. Must be from specified category.'''
            return not any([any([bad in f for f in c[:2]]) for bad in ",", "|"]) and c[0] == self.category_chooser.get_value()

        self.set_choices([self.make_measurement_choice(c[0], c[1])
                          for c in columns if valid_mc(c)])

