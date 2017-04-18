'''Classify - train or use a classifier'''

import bisect
import hashlib
import h5py
import logging
import numpy as np
import os
import pickle
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import RandomizedPCA
import tempfile

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps

logger = logging.getLogger(__name__)

USE_DOT = True
DEFAULT_N_ESTIMATORS = 25
DEFAULT_MIN_SAMPLES_PER_LEAF = 10
DEFAULT_RADIUS = 9
DEFAULT_N_FEATURES = 100

MODE_CLASSIFY = "Classify"
MODE_TRAIN = "Train"

AA_ADVANCED = "Advanced"
AA_AUTOMATIC = "Automatic"

G_TRAINING_SET = "TrainingSet"
DS_KERNEL = "Kernel"
DS_IMAGE_NUMBER = "ImageNumber"
DS_COORDS = "Coords"
A_VERSION = "Version"
A_CLASS = 'Class'
A_DIGEST = 'MD5Digest'
G_SAMPLING = "Sampling"
G_FILTERS = "Filters"
G_CLASSIFIERS = "Classifiers"
G_IMAGES = "Images"

CLS_GROUND_TRUTH = "GroundTruth"
CLS_KERNEL = "Kernel"
CLS_FILTER = "Filter"
CLS_CLASSIFIER = "Classifier"
CLS_SAMPLING = "Sampling"

SRC_OBJECTS = "Objects"
SRC_ILASTIK = "Ilastik"

ROUNDS = [("initial", 100000, 0),
          ("middle", 75000, 25000),
          ("final", 50000, 50000)]

class Classify(cpm.CPModule):
    variable_revision_number = 1
    category = "Image Processing"
    module_name = "Classify"
    def create_settings(self):
        self.mode = cps.Choice("Classify or train?", 
                               [MODE_CLASSIFY, MODE_TRAIN])
        self.advanced_or_automatic = cps.Choice(
            "Configuration mode", [AA_AUTOMATIC, AA_ADVANCED],
            doc = """Do you want to automatically choose the training parameters
            or use the defaults?""")
        self.radius = cps.Integer("Radius", DEFAULT_RADIUS, 1)
        self.n_features = cps.Integer(
            "Number of features", DEFAULT_N_FEATURES, 1,
            doc = """The classifier runs a feature reduction set. This creates
            <i>Eigentextures</i> which are representative texture patches
            found throughout the image. The module scores each patch around
            a pixel according to how much it has each of these textures and
            those scores are fed into the final classifier. Raise the number of
            features if some of the textures or edges of your classes are
            misclassified. Lower the number of features to improve processing
            time or to reduce overfitting if you have a smaller amount of
            ground truth.
            """)
        self.n_estimators = cps.Integer(
            "Number of estimators", DEFAULT_N_ESTIMATORS, 1,
            doc = """The classifier uses a voting scheme where it trains this
            many estimators. It purposefully does a bad job training and makes
            up for this deficit by having many poor classification judges.
            This protects against overfitting by not relying on having a single
            classifier that is very good at classifying the ground truth, but
            mistakenly uses irrelevant information to do so. Raise the number
            of estimators if the classifier is making obvious mistakes with
            unwarranted certainty. Lower the number of estimators to improve
            processing speed.""")
        self.min_samples_per_leaf = cps.Integer(
            "Minimum samples per leaf", DEFAULT_MIN_SAMPLES_PER_LEAF, 1,
            doc = """This setting determines the minimum number of ground truth
            pixels that the classifier will use to split a decision tree.
            There must be at least this number of example pixels in each branch
            for the classifier to have confidence that the split is real and
            not just an artifact of an irrelevant measurement.
            
            Lower this setting if the classifier does a good job on most of the
            pixels but does not draw sharp distinctions between one class and
            another at the border between the classes (e.g. at the edges of
            cells). Raise this setting if the classifier misclassifies pixels
            that are clearly not the right class - this is overtraining.
            """)
        self.path = cps.DirectoryPath("Classifier folder")
        def get_directory_fn():
            '''Get the directory for the file name'''
            return self.path.get_absolute_path()
        def set_directory_fn(path):
            dir_choice, custom_path = self.path.get_parts_from_path(path)
            self.path.join_parts(dir_choice, custom_path)
        self.filename = cps.FilenameText(
            "Classifier file", "Classifier.cpclassifier",
            get_directory_fn = get_directory_fn,
            set_directory_fn = set_directory_fn,
            exts = [("Pixel classifier (*.cpclassifier)", "*.cpclassifier"),
                    ("All files (*.*)", "*.*")])
        self.gt_source = cps.Choice(
            "Ground truth source", [SRC_OBJECTS, SRC_ILASTIK],
        doc="""
        The ground truth data can either be taken from objects or can be
        the exported TIF "labels" output of Ilastik.
        """)
        self.labels_image = cps.ImageNameSubscriber(
            "Ilastik labels image", "labels.tif",
            doc="""
            <i>Used only if the ground truth source is "Ilastik"</i>
            <br>
            This image should be the exported labels image from Ilastik.
            """)
        self.wants_background_class = cps.Binary(
            "Do you want a background class?", True)
        self.background_class_name = cps.Text("Background class name", 
                                              "Background")
        self.object_classes = []
        self.object_class_count = cps.HiddenCount(self.object_classes,
                                                  "Object class count")
        self.add_objects(False)
        self.add_objects_button = cps.DoSomething(
            "Add another class", "Add", self.add_objects)
        self.label_classes = []
        self.label_class_count = cps.HiddenCount(
            self.label_classes, "Label class count")
        self.add_labels(False)
        self.add_labels_button = cps.DoSomething(
        "Add another class", "Add", self.add_labels)
        
        self.images = []
        self.image_count = cps.HiddenCount(self.images, "Image count")
        self.add_image(False)
        self.add_image_button = cps.DoSomething(
            "Add another image", "Add", self.add_image)
        self.outputs = []
        self.output_count = cps.HiddenCount(self.outputs, "Output count")
        self.add_output(False)
        self.add_output_button = cps.DoSomething(
            "Add another output", "Add", self.add_output)
    
    def get_class_names(self, ignore=None):
        result = []
        if self.mode == MODE_TRAIN:
            if self.gt_source == SRC_OBJECTS:
                if self.wants_background_class:
                    result.append(self.background_class_name.value)
                result += [group.object_name.value 
                           for group in self.object_classes]
            else:
                result += [group.class_name.value
                           for group in self.label_classes]
        else:
            try:
                with self.get_classifier("r") as c:
                    return c.get_class_names()
            except IOError:
                result.append("None")
        return result
    
    def add_objects(self, can_remove=True):
        group = cps.SettingsGroup()
        group.append("object_name",  cps.ObjectNameSubscriber(
            "Object name", "Nuclei"))
        if can_remove:
            group.append("remover", cps.RemoveSettingButton(
                "Remove object", "Remove", self.object_classes, group))
        self.object_classes.append(group)
        
    def add_labels(self, can_remove=True):
        group = cps.SettingsGroup()
        group.append("class_name", cps.AlphanumericText(
            "Class name", "Class %d" % (len(self.label_classes)+1),
            doc="""
            The name to give to pixels of this class (e.g. "Foreground")
            
            You should add one class for each class you defined in Ilastik"""))
        if can_remove:
            group.append("remover", cps.RemoveSettingButton(
                "Remove object", "Remove", self.label_classes, group))
        self.label_classes.append(group)
        
    def add_image(self, can_remove=True):
        group = cps.SettingsGroup()
        group.append("image_name", cps.ImageNameSubscriber(
            "Image name", "DNA"))
        if can_remove:
            group.append("remover", cps.RemoveSettingButton(
                "Remove object", "Remove", self.images, group))
        self.images.append(group)
        
    def add_output(self, can_remove=True):
        group = cps.SettingsGroup()
        group.append(
            "output_image",
            cps.ImageNameProvider("Output image", "Probability"))
                
        group.append("class_name", cps.Choice(
            "Class name", choices=self.get_class_names(),
            choices_fn = self.get_class_names))
        if can_remove:
            group.append("remover", cps.RemoveSettingButton(
                "Remove object", "Remove", self.outputs, group))
        self.outputs.append(group)
        
    def settings(self):
        result = [
            self.object_class_count, self.label_class_count,
            self.image_count, self.output_count,
            self.mode, self.path, self.filename, 
            self.advanced_or_automatic,
            self.radius, self.n_features, self.n_estimators, 
            self.min_samples_per_leaf, self.gt_source,
            self.labels_image,
            self.wants_background_class, self.background_class_name]
        for group in self.object_classes:
            result += group.pipeline_settings()
        for group in self.label_classes:
            result += group.pipeline_settings()
        for group in self.images:
            result += group.pipeline_settings()
        for group in self.outputs:
            result += group.pipeline_settings()
        return result
            
    def visible_settings(self):
        result = [self.mode, self.path, self.filename]
        if self.mode == MODE_TRAIN:
            result.append(self.advanced_or_automatic)
            if self.advanced_or_automatic == AA_ADVANCED:
                result += [self.radius, self.n_features, self.n_estimators,
                           self.min_samples_per_leaf]
        for group in self.images:
            result += group.visible_settings()
        result.append(self.add_image_button)
        if self.mode == MODE_TRAIN:
            self.filename.mode = cps.FilenameText.MODE_OVERWRITE
            result.append(self.gt_source)
            if self.gt_source == SRC_OBJECTS:
                result.append(self.wants_background_class)
                if self.wants_background_class:
                    result.append(self.background_class_name)
                for group in self.object_classes:
                    result += group.visible_settings()
                result.append(self.add_objects_button)
            else:
                result.append(self.labels_image)
                for group in self.label_classes:
                    result += group.visible_settings()
                result.append(self.add_labels_button)
        else:
            self.filename.mode = cps.FilenameText.MODE_OPEN
            for group in self.outputs:
                result += group.visible_settings()
            result.append(self.add_output_button)
        return result

    def prepare_settings(self, setting_values):
        for count, sequence, add_fn in zip(
            [int(_) for _ in setting_values[:4]],
            (self.object_classes, self.label_classes, self.images, self.outputs),
            (self.add_objects, self.add_labels, self.add_image, self.add_output)):
            del sequence[:]
            for idx in range(count):
                add_fn()

    def is_aggregation_module(self):
        return self.mode == MODE_TRAIN

    def get_classifier(self, mode):
        path = os.path.join(self.path.get_absolute_path(), self.filename.value)
        return PixelClassifier(path, mode)
    
    def get_radius(self):
        if self.advanced_or_automatic == AA_AUTOMATIC:
            return DEFAULT_RADIUS
        return self.radius.value
    
    def get_n_features(self):
        if self.advanced_or_automatic == AA_AUTOMATIC:
            return DEFAULT_N_FEATURES
        return self.n_features.value
    
    def get_n_estimators(self):
        if self.advanced_or_automatic == AA_AUTOMATIC:
            return DEFAULT_N_ESTIMATORS
        return self.n_estimators.value
    
    def get_min_samples_per_leaf(self):
        if self.advanced_or_automatic == AA_AUTOMATIC:
            return DEFAULT_MIN_SAMPLES_PER_LEAF
        return self.min_samples_per_leaf.value
    
    def prepare_group(self, workspace, grouping, image_numbers):
        if self.mode == MODE_TRAIN:
            with self.get_classifier("w") as c:
                assert isinstance(c, PixelClassifier)
                r = self.get_radius()
                i, j = np.mgrid[-r:r+1, -r:r+1]
                kernel_mask = i*i + j*j <= r*r
                n_features = np.sum(kernel_mask)
                kernel = np.vstack([
                    np.column_stack([
                        np.ones(n_features, int) * channel, # C offset
                        np.zeros(n_features, int), # T offset
                        np.zeros(n_features, int), # Z offset
                        i[kernel_mask],
                        j[kernel_mask]])
                    for channel in range(len(self.images))])
                c.set_kernel(kernel)
                for class_name in self.get_class_names():
                    c.add_class(class_name)
                    
    def run(self, workspace):
        if self.mode == MODE_TRAIN:
            self.run_train(workspace)
        else:
            self.run_classify(workspace)
   
    def get_5d_image(self, workspace):
        '''Compile a 5d image from the channel planes'''
        pixels = []
        for group in self.images:
            image_name = group.image_name.value
            img = workspace.image_set.get_image(
                image_name, must_be_grayscale=True).pixel_data
            #
            # [[img]] adds Z and T of dimension 1 to the uber array
            #
            pixels.append([[img]])
        return np.array(pixels)
        
    def run_train(self, workspace):
        pixels = self.get_5d_image(workspace)
        image_number = workspace.measurements.image_number
        with self.get_classifier("a") as c:
            assert isinstance(c, PixelClassifier)
            c.add_image(pixels, image_number)
            gt = []
            if self.gt_source == SRC_OBJECTS:
                bg = np.ones(pixels.shape[-2:], bool)
                for group in self.object_classes:
                    object_name = group.object_name.value
                    objects = workspace.object_set.get_objects(object_name)
                    fg = np.zeros(bg.shape, bool)
                    for plane, _ in objects.get_labels():
                        fg[plane > 0] = True
                    bg[fg] = False
                    gt.append((object_name, fg))
                if self.wants_background_class:
                    gt.append((self.background_class_name.value, bg))
            else:
                label_image_name = self.labels_image.value
                img = workspace.image_set.get_image(label_image_name)
                pixel_data = (img.pixel_data * img.scale).astype(int)
                for idx, group in enumerate(self.label_classes):
                    class_name = group.class_name.value
                    gt.append((class_name, pixel_data == idx+1))
            for object_name, fg in gt:
                i, j = np.where(fg)
                c.add_ground_truth(
                    object_name,
                    image_number,
                    np.column_stack([np.zeros(len(i), int)] * 3 + [i, j]))
    
    def post_group(self, workspace, grouping):
        if self.mode == MODE_TRAIN:
            last_round_name = None
            for round_name, n_random, n_error in ROUNDS:
                self.do_training_round(
                    last_round_name, round_name,
                    n_random / len(self.get_class_names()),
                    n_error / len(self.get_class_names()))
            with self.get_classifier("a") as c:
                assert isinstance(c, PixelClassifier)
                c.config_final_pipeline("final", "final")
    
    def do_training_round(
        self, name_in, name_out, n_random_samples, n_error_samples):
        '''Perform a round of training
        
        name_in - name of the filter bank and classifier to use to find the
                  error samples. None if no error samples.
        name_out - name for output classifier
        n_random_samples - # of samples randomly drawn from each class
        n_error_samples - # of samples drawn from errors
        '''
        with self.get_classifier("a") as c:
            assert isinstance(c, PixelClassifier)
            #
            # Sample
            #
            fb_sample_name = name_out + "_filter_bank"
            classifier_sample_name = name_out + "_classifier"
            for sample_name in fb_sample_name, classifier_sample_name:
                d = {}
                for idx, class_name in enumerate(c.get_class_names()):
                    gt = c.get_ground_truth(class_name)
                    if name_in is not None and n_error_samples > 0:
                        probs = c.run_pipeline(name_in, name_in, gt)[:, idx]
                        order = np.argsort(probs)
                        error_idx = order[:n_error_samples]
                        other_idx = order[n_error_samples:]
                    else:
                        error_idx = np.zeros(0, int)
                        other_idx = np.arange(gt.shape[0])
                    if len(other_idx) > n_random_samples:
                        r = c.random_state(str(name_in)+name_out+class_name)
                        other_idx = r.choice(
                            other_idx, size=n_random_samples, replace=False)
                    sample_idx = np.hstack((error_idx, other_idx))
                    d[class_name] = sample_idx
                c.add_sampling(sample_name, d)
            samples, classes = c.sample(fb_sample_name)
            c.make_filter_bank(
                samples, classes, name_out, self.get_n_features())
            samples, classes = c.sample(classifier_sample_name)
            filtered = c.use_filter_bank(name_out, samples)
            algorithm = ExtraTreesClassifier(
                n_estimators = self.get_n_estimators(),
                min_samples_leaf=self.get_min_samples_per_leaf())
            c.fit(name_out, filtered, classes, algorithm)
        
    def run_classify(self, workspace):
        pixels = self.get_5d_image(workspace)
        #
        # Process the image in chunks
        #
        with self.get_classifier("r") as c:
            assert isinstance(c, PixelClassifier)
            class_names = c.get_class_names()
            prob_idxs = np.array([
                class_names.index(group.class_name.value)
                for group in self.outputs])
            chunk_size = 128
            prob_maps = np.zeros(
                (len(prob_idxs), pixels.shape[3], pixels.shape[4]))
            for i in range(0, pixels.shape[3], chunk_size):
                iend = min(i+chunk_size, pixels.shape[3])
                for j in range(0, pixels.shape[4], chunk_size):
                    jend = min(j+chunk_size, pixels.shape[4])
                    ii, jj = [_.flatten() for _ in np.mgrid[i:iend, j:jend]]
                    coords = np.column_stack((
                        np.zeros(len(ii), int),
                        np.zeros(len(ii), int),
                        np.zeros(len(ii), int),
                        ii, jj))
                    samples = c.get_samples(pixels, coords)
                    probs = c.run_final_pipeline(samples)
                    prob_maps[:, i:iend, j:jend] = \
                        probs[:, prob_idxs].reshape(
                            iend - i, jend - j, len(prob_maps)).transpose(2, 0, 1)
        for i, group in enumerate(self.outputs):
            image_name = group.output_image.value
            image = cpi.Image(prob_maps[i])
            workspace.image_set.add(image_name, image)
        if self.show_window:
            workspace.display_data.input_images = [
                pixels[i].reshape(*pixels.shape[-2:]) 
                for i in range(pixels.shape[0])]
            workspace.display_data.output_images = [
                prob_maps[i] for i in range(len(prob_maps))]
            
    def display(self, workspace, figure):
        if self.mode == MODE_CLASSIFY:
            figure.set_subplots((2, max(len(self.images), len(self.outputs))))
            for i, (group, image) in enumerate(
                zip(self.images, workspace.display_data.input_images)):
                figure.subplot_imshow_bw(0, i, image,
                                         title = group.image_name.value)
            for i, (group, image) in enumerate(
                zip(self.outputs, workspace.display_data.output_images)):
                figure.subplot_imshow_bw(1, i, image,
                                         title = group.output_image.value)
            
            
#
# The classifier class is here mostly to simplify the initial development
# and it would get moved out to Centrosome eventually.
#
class PixelClassifier(object):
    '''Represents a classifier stored in an HDF5 file
    
    The parts:
    
    Kernel - this is the patch that's taken from the pixel's neighborhood. It
             has the shape, NxM where N is the # of points in the patch and
             M are the indices relative to the pixel. The kernel dimensions
             are C, T, Z, Y, X.
    TrainingSet - the ground truth data. Each label has a corresponding
             dataset stored using the name of the class. The dataset has the
             shape, S x N, where N is the # of points in the patch and S is
             the # of samples for that class.
    Filters - the filters applied to the kernel to derive features. Each
              filter is a vector of length N.
    Classifier - the classifier is pickled after being trained and is stored
                 in a dataset.
    '''
    version = 1
    def __init__(self, path, mode, classifier_path = None):
        '''Either create or load the classifier from a file
        
        path - path to the file
        mode - "r" for read-only, "w" for overwrite (new file) or "a" for
               read-write access
        classifier_path - path to the sub-groups within the HDF file, defaults
                          to the root.
        '''
        self.f = h5py.File(path, mode)
        self.root = \
            self.f if classifier_path is None else self.f[classifier_path]
        if mode == "w":
            self.f.attrs[A_VERSION] = self.version
            self.g_training_set = self.root.create_group(G_TRAINING_SET)
            self.g_filters = self.root.create_group(G_FILTERS)
            self.g_sampling = self.root.create_group(G_SAMPLING)
            self.g_classifiers = self.root.create_group(G_CLASSIFIERS)
            self.g_images = self.root.create_group(G_IMAGES)
        else:
            self.g_training_set = self.root[G_TRAINING_SET]
            self.g_filters = self.root[G_FILTERS]
            self.g_sampling = self.root[G_SAMPLING]
            self.g_classifiers = self.root[G_CLASSIFIERS]
            self.g_images = self.root[G_IMAGES]
        self.classifier_cache = {}
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.f.flush()
        del self.root
        del self.g_training_set
        del self.g_filters
        del self.g_classifiers
        del self.g_sampling
        self.f.close()
        del self.f
        
    def random_state(self, extra = ''):
        '''Return a random state based on the ground truth sampled
    
        '''
        if A_DIGEST not in self.g_training_set.attrs:
            md5 = hashlib.md5()
            for class_name in self.get_class_names():
                md5.update(self.get_ground_truth(class_name).value.data)
            self.g_training_set.attrs[A_DIGEST] = md5.hexdigest()
        return np.random.RandomState(np.frombuffer(
            self.g_training_set.attrs[A_DIGEST] + extra, np.uint8))
    
    @staticmethod
    def get_instances(group, class_name):
        '''Get the keys for a group for objects of a given class'''
        return sorted([k for k in group.keys()
                       if group[k].attrs[A_CLASS] == class_name])        

    def set_kernel(self, kernel):
        '''Set the kernel used to sample pixels from a neighborhood
        
        kernel - an N x 5 matrix of N offsets by 5 dimensions (C, T, Z, Y, X)
        '''
        if DS_KERNEL in self.root.keys():
            del self.root[DS_KERNEL]
        ds = self.root.create_dataset(DS_KERNEL, data=kernel)
        ds.attrs[A_CLASS] = CLS_KERNEL
        
    def get_kernel(self):
        return self.root[DS_KERNEL][:]
    
    def add_class(self, class_name):
        ds = self.g_training_set.create_dataset(
            class_name,
            shape = (0, 6),
            dtype = np.int32,
            chunks = (4096, 6),
            maxshape = (None, 6))
        ds.attrs[A_CLASS] = CLS_GROUND_TRUTH
        
    def get_class_names(self):
        '''Get the names of the classifier's classes'''
        return self.get_instances(self.g_training_set, CLS_GROUND_TRUTH)
        
    def get_ground_truth(self, class_name):
        '''Get the ground truth for a class
        
        class_name - the name of the class
        
        returns an S X 6 where the first index is the image number and
        the remaining are the coordinates of the GT pixel in C, T, Z, Y, X form
        '''
        return self.g_training_set[class_name]
    
    @property
    def gt_chunk_size(self):
        '''The size of a chunk of ground truth that fits in memory'''
        kernel_size = self.get_kernel().shape[0]
        chunk_size = int(50*1000*1000 / kernel_size)
        return chunk_size
    
    @property
    def pix_chunk_size(self):
        return 1000 * 1000
    
    def add_image(self, image, image_number):
        image_number = str(image_number)
        if image_number in self.g_images.keys():
            del self.g_images[image_number]
        self.g_images.create_dataset(image_number, data = image)
        
    def get_image(self, image_number):
        image_number = str(image_number)
        return self.g_images[image_number].value
        
    def add_ground_truth(self, class_name, image_number, coordinates):
        '''Add ground truth to a class
        
        class_name - name of the class
        image_number - the image number as reported in add_image
        pixels - an S x 5 matrix of S samples and 5 pixel coordinates
        '''
        coordinates = np.column_stack(
            (np.ones(coordinates.shape[0], coordinates.dtype) * image_number,
             coordinates))
        ds = self.get_ground_truth(class_name)
        ds_idx = ds.shape[0]
        ds.resize(ds_idx + coordinates.shape[0], axis = 0)
        ds[ds_idx:] = coordinates
        
        if A_DIGEST in self.g_training_set.attrs:
            del self.g_training_set.attrs[A_DIGEST]
        
    def get_samples(self, image, pixels):
        '''Extract samples from an image at given pixels
        
        image - a C, T, Z, Y, X image
        
        pixels - an S x 5 matrix where the columns are the C, T, Z, Y, X
                 coordinates and the rows are the samples to collect
        
        returns an S x N matrix where N is the size of the kernel
        '''
        kernel = self.get_kernel()[np.newaxis, :, :]
        coords = pixels[:, np.newaxis, :] + kernel
        #
        # Boundary reflection
        #
        coords[coords < 0] = np.abs(coords[coords < 0])
        for i, axis_size in enumerate(image.shape):
            mask = coords[:, :, i] >= axis_size
            coords[mask, i] = axis_size * 2 - coords[mask, i] - 1
        #
        # Samples
        #
        samples = image[coords[:, :, 0],
                        coords[:, :, 1],
                        coords[:, :, 2],
                        coords[:, :, 3],
                        coords[:, :, 4]]
        return samples
    
    def add_sampling(self, sampling_name, d_index):
        '''Add a sampling of the ground truth
        
        sampling_name - a name for the sampling
        d_index - a dictionary of indices. The key is the class name and
                  the value is a vector of indices into the class's ground truth
        '''
        if sampling_name in self.g_sampling.keys():
            del self.g_sampling[sampling_name]
        g = self.g_sampling.create_group(sampling_name)
        for k, v in d_index.iteritems():
            g.create_dataset(k, data=v)
        
    def sample(self, sampling_name):
        '''Return sample and vector of class indexes
        
        sampling_name - the name of the sampling
        
        Returns a sample which is S x N and vector of length S which is
        composed of indexes into the class names returned by get_class_names.
        S is the length of the sum of all samples in the sampling
        '''
        g = self.g_sampling[sampling_name]
        samples = []
        classes = []
        for idx, class_name in enumerate(self.get_class_names()):
            if class_name in g.keys():
                sampling = g[class_name][:]
                classes.append(np.ones(len(sampling), np.uint8) * idx)
                gt = self.get_ground_truth(class_name)
                #
                # h5py datasets are not addressable via an array of indices
                # in the way that numpy arrays are. A mask of the array elements
                # to be processed is handled by a loop through each selected
                # element; it takes ~ hours to process arrays of our size. The
                # ground truth may be too large to bring into memory as a
                # Numpy array.
                #
                # We sort the sampling indices and then process in chunks.
                #
                if len(gt) == len(sampling):
                    logger.debug("Extracting %d samples from %s" % 
                                 (len(sampling), class_name))
                    samples.append(gt[:])
                else:
                    chunk_size = self.pix_chunk_size
                    sampling.sort()
                    sindx = 0
                    for gtidx in range(0, len(gt), chunk_size):
                        gtidx_end = min(gtidx+chunk_size, len(gt))
                        if sampling[sindx] >= gtidx_end:
                            continue
                        sindx_end = bisect.bisect_left(
                            sampling[sindx:], gtidx_end) + sindx
                        logger.debug(
                            "Extracting %d samples from %s %d:%d" %
                            (sindx_end-sindx, class_name, gtidx, gtidx_end))
                        samples.append(gt[:][sampling[sindx:sindx_end], :])
                        sindx = sindx_end
                        if sindx >= len(gt):
                            break
        samples = np.vstack(samples)
        classes = np.hstack(classes)
        #
        # Order by image number.
        #
        order = np.lexsort(
            [samples[:, _] for _ in reversed(range(samples.shape[1]))])
        samples = samples[order]
        classes = classes[order]
        counts = np.bincount(samples[:, 0])
        image_numbers = np.where(counts > 0)[0]
        counts = counts[image_numbers]
        idxs = np.hstack([[0], np.cumsum(counts)])
        result = []
        for image_number, idx, idx_end in zip(
            image_numbers, idxs[:-1], idxs[1:]):
            image = self.get_image(image_number)
            result.append(self.get_samples(image, samples[idx:idx_end, 1:]))
        
        return np.vstack(result), classes
    
    def make_filter_bank(self, sampling, classes, filter_bank_name, n_filters, 
                         algorithm=None):
        '''Make a filter bank using PCA
        
        sampling - a sampling of the ground truth
        classes - a vector of the same length as the sampling giving the
                  indexes of the classes of each sample
        filter_bank_name - the name to assign to the filter bank
        n_filters - # of filters to create
        
        algorithm - an object that can be fitted using algorithm.fit(X, Y)
                    and can transform using algorithm.transform(X). Default
                    is RandomizedPCA.
        '''
        if algorithm is None:
            r = self.random_state(filter_bank_name)
            algorithm = RandomizedPCA(n_filters,
                                      random_state = r)
        algorithm.fit(sampling, classes)
        if hasattr(algorithm, "components_"):
            components = algorithm.components_
            if len(components) > n_filters:
                components = components[:n_filters]
            if filter_bank_name in self.g_filters.keys():
                del self.g_filters[filter_bank_name]
            ds = self.g_filters.create_dataset(filter_bank_name,
                                               data = components)
            ds.attrs[A_CLASS] = CLS_FILTER
        else:
            s = pickle.dumps(algorithm)
            ds = self.g_filters.create_dataset(filter_bank_name,
                                               data = s)
            ds.attrs[A_CLASS] = CLS_CLASSIFIER
            
    def use_filter_bank(self, filter_bank_name, sample):
        '''Transform a sample using a filter bank'''
        ds = self.g_filters[filter_bank_name]
        if ds.attrs[A_CLASS] == CLS_FILTER:
            if USE_DOT:
                result = np.dot(sample, ds[:].T)
            else:
                #
                # A dot product... but cluster's np.dot is so XXXXed
                #
                chunk_size = self.gt_chunk_size
                result = []
                for idx in range(0, len(sample), chunk_size):
                    idx_end = min(idx + chunk_size, len(sample))
                    logger.debug("Processing dot product chunk %d:%d of %d" %
                                 (idx, idx_end, len(sample)))
                    result.append(np.sum(sample[idx:idx_end, :, np.newaxis] * 
                                         ds[:].T[np.newaxis, :, :], 1))
                result = np.vstack(result)
        else:
            algorithm = pickle.loads(ds.value)
            result = algorithm.transform(sample)
        return result
    
    def fit(self, classifier_name, sample, classes, algorithm=None):
        '''Fit an algorithm to data and save
        
        classifier_name - save using this name
        sample - S samples x N features
        classes - S class labels indexing into the class names
        algorithm - algorithm to use to train
        '''
        if algorithm is None:
            algorithm = ExtraTreesClassifier(
                n_estimators=N_ESTIMATORS,
                min_samples_leaf = MIN_SAMPLES_PER_LEAF)
        algorithm.fit(sample, classes)
        s = pickle.dumps(algorithm)
        if classifier_name in self.g_classifiers.keys():
            del self.g_classifiers[classifier_name]
        ds = self.g_classifiers.create_dataset(classifier_name, data = s)
        ds.attrs[A_CLASS] = CLS_CLASSIFIER
        
    def predict_proba(self, classifier_name, sample):
        if classifier_name not in self.classifier_cache:
            algorithm = pickle.loads(self.g_classifiers[classifier_name].value)
            self.classifier_cache[classifier_name] = algorithm
        else:
            algorithm = self.classifier_cache[classifier_name]
        return algorithm.predict_proba(sample)
    
    def run_pipeline(self, filter_bank_name, classifier_name, sample):
        filtered = self.use_filter_bank(filter_bank_name, sample)
        return self.predict_proba(classifier_name, filtered)
    
    def config_final_pipeline(self, filter_bank_name, classifier_name):
        self.root.attrs["FilterBankName"] = filter_bank_name
        self.root.attrs["ClassifierName"] = classifier_name
        
    def run_final_pipeline(self, sample):
        return self.run_pipeline(self.root.attrs["FilterBankName"],
                                 self.root.attrs["ClassifierName"],
                                 sample)
    
if __name__ == "__main__":
    logging.basicConfig()
    logging.root.setLevel(logging.DEBUG)
    import cellprofiler.pipeline as cpp
    import cellprofiler.preferences as cpprefs
    cpprefs.set_default_output_directory("c:/temp/output/classify")
    pipeline = cpp.Pipeline()
    pipeline.load("c:/temp/output/classify/classify.cpproj")
    module = pipeline.modules()[-1]
    module.post_group(None, None)
    from cellprofiler.utilities.cpjvm import cp_stop_vm
    cp_stop_vm()
