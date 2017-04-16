
MEAS_TRAM = "TrAM"
CAT_TRAM = "TRAM"
IMAGE = "Image"

# todo
__doc__ = ""

import logging

logger = logging.getLogger(__name__)

import numpy as np
import itertools as it
import cellprofiler.module as cpm
import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
import cellprofiler.measurement as cpmeas
import cellprofiler.preferences as cpprefs



class TrAM(cpm.Module):
    module_name = "TrAM"
    category = "Object Processing" # todo: is this right?
    variable_revision_number = 1

    def create_settings(self): # todo: 1st draft
        # for them to choose the tracked objects
        self.object_name = cps.ObjectNameSubscriber(
                "Select the tracked objects", cps.NONE, doc="""
            Select the tracked objects for computing TrAM.""")

        # which measurements will go into the TrAM computation
        self.tram_measurements = cps.MeasurementMultiChoice(
            "TrAM measurements", doc="""
            This setting defines the tracked quantities that will be used
            to compute the TrAM metric. At least one must be selected.""")

        # spline knots
        self.num_knots = cps.Integer(
            "Number of spline knots", 4, minval=3, doc="""
            Number of knots to use in the spline fit to time series""")

        # TrAM exponent
        self.p = cps.Float(
            "TrAM exponent", 0.5, minval=0.01, maxval=1, doc="""
            The exponent used to combine different measurements.""")

    def settings(self): # todo: 1st draft
        return [self.object_name, self.tram_measurements, self.num_knots, self.p]

    def validate_module(self, pipeline): # todo: 1st draft
        '''Make sure that the user has selected at least one measurement for TrAM'''
        if (len(self.get_selected_tram_measurements()) == 0):
            raise cps.ValidationError(
                    "Please select at least one TrAM measurement for tracking of " + self.object_name.get_value(),
                    self.tram_measurements)

    def visible_settings(self): # todo: 1st draft
        return self.settings()

    def prepare_group(self, workspace, grouping, image_numbers): #todo: need to do anything?
        pass

    def run(self, workspace): #todo - is there anything we need to do here?
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
       
        pass

    def display(self, workspace, figure): #todo
        """

        pass


    def post_group(self, workspace, grouping): #todo: main routine. In progress
        pipeline = workspace.pipeline
        image_set = workspace.image_set
        object_set = workspace.object_set
        measurements = workspace.measurements
        frame = workspace.frame
        display_data = workspace.display_data

        obj_name = self.object_name.get_value()

        names = measurements.get_names()
        cols = measurements.get_measurement_columns()
        obj_names = measurements.get_object_names()
        group = measurements.get_group_number()

        # find the tracking label
        feature_names_list = [measurements.get_feature_names(obj_name) for obj_name in obj_names]
        feature_names = dict(zip(obj_names, feature_names_list))[obj_name]
        tracking_label_feature_name = [name for name in feature_names if name.startswith("TrackObjects_Label")][0]
        # and get its values
        label_vals = measurements.get_measurement(obj_name, tracking_label_feature_name, measurements.get_image_numbers())
        label_dict = {tracking_label_feature_name : label_vals} # keep it

        selections = self.get_selected_tram_measurements()

        # get a tuple dictionary entry relating feature name with data values
        def get_dict_entry(sel):
            feat_obj_name, feat_name = sel.split("|")
            vals = measurements.get_measurement(feat_obj_name, feat_name, measurements.get_image_numbers())
            return (feat_name, vals)

        dict_entry_list = [get_dict_entry(sel) for sel in selections]
        all_values_dict = dict(dict_entry_list) # contains dictionary from feature name to values
        all_values_dict.update(label_dict) # add the label dictionary

        img_numbers = measurements.get_image_numbers()

        # get image numbers into the dict
        counts = [len(x) for x in label_vals] # number of tracked objects at each time point
        z = zip(img_numbers, counts)
        image_vals = [[image for _ in xrange(count)] for image, count in zip(img_numbers, counts)]
        all_values_dict.update({IMAGE: image_vals})

        # We have all the data in lists of lists. Flatten and make data frame.
        def flatten_list_of_lists(lol):
            return list(it.chain.from_iterable(lol))
        vector_dict = {k : flatten_list_of_lists(v) for k, v in all_values_dict.items()}

        # create dictionary that doesn't have the tracking label or image number
        data_only_dict = vector_dict.copy()
        label_vec = data_only_dict.pop(tracking_label_feature_name)
        image_vec = data_only_dict.pop(IMAGE)


        # todo: make data frame if pandas ok to use
        tad = TrAM.compute_typical_deviations(data_only_dict, label_vec, image_vec)

        # put all the data into a 2D array and normalize by typical deviations
        data_array = np.column_stack(data_only_dict.values())
        data_keys = data_only_dict.keys()
        inv_devs = np.diag([1/tad[k] for k in data_keys])
        normalized_data_array = np.dot(data_array, inv_devs)

        # now get all unique nucleus labels and run through them, computing tram
        tram_dict = dict()
        for label in set(label_vec):
            indexes = [i for i in range(0, len(label_vec)) if label_vec[i] == label]

            if len(indexes) < 6: # todo figure this out a better way
                tram = float('nan')
            else:
                normalized_data_for_label = normalized_data_array[indexes,] # get the corresponding data
                images = [image_vec[i] for i in indexes]
                normalized_data_for_label = normalized_data_for_label[np.argsort(images),] # order by time
                normalized_data_dict = {data_keys[i] : normalized_data_for_label[:,i] for i in range(0,len(data_keys)) }
                tram = self.compute_TrAM(normalized_data_dict)
            tram_dict.update({label : tram})

        pass

    def compute_TrAM(self, normalized_values_dict): # todo
        """
        
        :param normalized_values_dict: keys are feature names, values are normalized feature values across the track 
        :return: TrAM value for this trajectory
        """

        # todo: enforce Euclidian for XY data

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
        tram = np.max(np.power(np.mean(np.power(aberration_array, p), 1), 1.0/p)) # todo: double check formula

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

        # mapping from label to indexes
        labels_dict = dict()
        labels_set = set(labels_vec)
        for label in labels_set:
            indexes = np.flatnonzero(labels_vec == label) # which
            labels_dict.update({label : indexes})

        result = dict()
        # for each feature get the deltas in time
        for feat_name, values in values_dict.items():
            all_diffs = list()
            for label, indexes in labels_dict.items():
                data = [values[i] for i in indexes]
                images = [image_vec[i] for i in indexes]
                z = sorted(zip(images, data)) # get them in time order
                ordered_data = [data for _, data in z]
                all_diffs.append(ordered_data)
            mad = compute_median_abs_deviation(all_diffs)
            result.update({feat_name : mad})


        return result


    # Get the selected measurements, restricted to those which start with the object name
    def get_selected_tram_measurements(self):
        # get what was selected by the user
        selections = self.tram_measurements.get_selections()

        # get the object set to work on
        object_name = self.object_name.get_value()

        return [sel for sel in selections if sel.startswith(object_name)]


    def get_measurement_columns(self, pipeline): #todo: 1st draft. Need to verify correct
        return [self.object_name.get_value(), MEAS_TRAM, cpmeas.COLTYPE_FLOAT]

    def get_object_relationships(self, pipeline): #todo: 1st draft. Need to verify
        '''Return the object relationships produced by this module'''
        object_name = self.object_name.value

        return [("Parent", object_name, object_name, cpmeas.MCA_AVAILABLE_POST_GROUP)]


    def get_categories(self, pipeline, object_name): #todo.  1st draft need to verify
        if object_name ==self.object.name:
            return [CAT_TRAM]
        return []

    def get_measurements(self, pipeline, object_name, category): #todo: 1st draft
        if object_name == self.object_name.get_value() and category == CAT_TRAM:
            return [MEAS_TRAM]
        return []

    def get_measurement_objects(self, pipeline, object_name, category,
                                measurement): #todo: 1st draft
        if (object_name == self.object_name.get_value() and category == CAT_TRAM and
                    measurement == MEAS_TRAM):
            return [self.object_name.value]
        return []

    def get_measurement_scales(self, pipeline, object_name, category, feature, image_name): # todo: 1st draft
        return [self.p, self.num_knots]


    def is_aggregation_module(self): # todo - not sure what to return here
        """If true, the module uses data from other imagesets in a group

        Aggregation modules perform operations that require access to
        all image sets in a group, generally resulting in an aggregation
        operation during the last image set or in post_group. Examples are
        TrackObjects, MakeProjection and CorrectIllumination_Calculate.
        """
        return True