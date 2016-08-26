'''<b>MeasuremeRadialEntropy</b> - a module which does a thing
<hr>

'''
#################################
#
# Imports from useful Python libraries
#
#################################

import numpy as np
from scipy import stats

#################################
#
# Imports from CellProfiler
#
# The package aliases are the standard ones we use
# throughout the code.
#
##################################

import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.setting as cps

from centrosome.cpmorphology import minimum_enclosing_circle



ENTROPY = "Entropy"



class MeasurementTemplate(cpm.Module):

    module_name = "MeasureRadialEntropy"
    category = "Measurement"
    variable_revision_number = 1


    def create_settings(self):

        self.input_object_name = cps.ObjectNameSubscriber(
            "Select objects to measure",
            doc="""Select the objects whose radial entropy you want to measure.""")

        self.input_image_name = cps.ImageNameSubscriber(
            "Select an image to measure", doc="""Select the
            grayscale image you want to measure the entropy of.""")

        self.bin_number=cps.Integer(
            "Input number of bins", 6, minval=3, maxval=60,
            doc="""Number of radial bins to divide your object into""")

        self.intensity_measurement=cps.MultiChoice(
            "Which intensity measurement should be used?", ['Mean','Median'], value='Mean')

    def settings(self):
        return [self.input_image_name, self.input_object_name,
                self.intensity_measurement, self.bin_number]


    def run(self, workspace):
        meas = workspace.measurements
        assert isinstance(meas, cpmeas.Measurements)

        statistics = [["Entropy"]]

        workspace.display_data.statistics = statistics

        input_image_name = self.input_image_name.value
        input_object_name = self.input_object_name.value
        metric = self.intensity_measurement.value
        bins = self.bin_number.value

        image_set = workspace.image_set

        input_image = image_set.get_image(input_image_name,
                                          must_be_grayscale=True)

        pixels = input_image.pixel_data

        object_set = workspace.object_set
        assert isinstance(object_set, cpo.ObjectSet)

        objects = object_set.get_objects(input_object_name)
        labels = objects.segmented

        indexes = objects.indices

        centers, radius = minimum_enclosing_circle(labels, indexes)

        feature = self.get_measurement_name(input_image_name,metric,bins)
        entropy=self.slice_and_measure_intensity(pixels,labels,indexes,centers,metric,bins)

        meas.add_measurement(input_object_name,feature,entropy)

        emean = np.mean(entropy)
        statistics.append([feature, emean])


    ################################
    #
    # DISPLAY
    #
    def display(self, workspace, figure=None):
        statistics = workspace.display_data.statistics
        if figure is None:
            figure = workspace.create_or_find_figure(subplots=(1, 1,))
        else:
            figure.set_subplots((1, 1))
        figure.subplot_table(0, 0, statistics)


    ############################################
    #
    #
    #

    def slice_and_measure_intensity(self, pixels, labels, indexes,centers,metric,nbins):
        entropylist=[]
        for eachindex in indexes:
            objects = np.zeros_like(pixels)
            objects[objects==0]=-1
            objects[labels.segmented==eachindex]= pixels[labels.segmented==eachindex]
            pixeldict={}
            for i1,i2 in objects.shape:
                if objects[i1,i2]!= -1:
                    center_x = centers[indexes.index(eachindex), 1]
                    center_y = centers[indexes.index(eachindex), 0]
                    sliceno = np.int32((np.pi + np.arctan2(i1-center_y, i2-center_x)) * (nbins / (2 * np.pi)))
                    if sliceno not in pixeldict.keys():
                        pixeldict[sliceno]=[objects[i1,i2]]
                    else:
                        pixeldict[sliceno] += [objects[i1, i2]]
            entropy=calculate_entropy(pixeldict,metric)
            entropylist.append(entropy)
        entropyarray=np.array(entropylist)
        return entropyarray

    def calculate_entropy(pixeldict,metric):
        slicemeasurements=[]
        for eachslice in pixeldict.keys():
            if metric=='mean':
                slicemeasurements.append(np.mean(pixeldict[eachslice]))
            else:
                slicemeasurements.append(np.median(pixeldict[eachslice]))
        slicemeasurements=np.array(slicemeasurements,dtype=float)
        slicemeasurements=slicemeasurements/sum(slicemeasurements)
        entropy=stats.entropy(slicemeasurements)
        return entropy


    def get_feature_name(self,input_image_name,metric,bins):
        '''Return a measurement feature name for the given Zernike'''
        #
        # Something nice and simple for a name... Intensity_DNA_N4M2 for instance
        #
        return "%s_%s_%d" % (input_image_name, metric, bins)

    def get_measurement_name(self, input_image_name, metric, bins):
        '''Return the whole measurement name'''
        return '_'.join([ENTROPY,
                         self.get_feature_name(self,input_image_name,metric,bins)])


    def get_measurement_columns(self, pipeline):

        input_object_name = self.input_object_name.value

        input_image_name=self.input_image_name.value
        metric = self.intensity_measurement.value
        bins = self.bin_number.value

        return [(input_object_name,
                 self.get_measurement_name(input_image_name,metric,bins),
                 cpmeas.COLTYPE_FLOAT)]


    def get_categories(self, pipeline, object_name):
        if object_name == self.input_object_name:
            return [ENTROPY]
        else:
            # Don't forget to return SOMETHING! I do this all the time
            # and CP mysteriously bombs when you use ImageMath
            return []


    def get_measurements(self, pipeline, object_name, category):
        if (object_name == self.input_object_name and
                    category == ENTROPY):
            return ["Entropy"]
        else:
            return []




    @staticmethod


'''Idea is
Get label matrix for objects
Take one object
Divide the points into n wedges using the formula from stack overflow
Use mean or median (as instructed) on values for each wedge
Divide by summation to get to equal 1
Calculate entropy with scipystats, return
sliceno = numpy.int32((pi + numpy.arctan2(Y, X)) * (N / (2*pi)))
meaning:

compute the angle -pi...pi for each point with arctan2
shift by pi to make it a positive interval
rescale to 0..N-1
convert to an integer

'''