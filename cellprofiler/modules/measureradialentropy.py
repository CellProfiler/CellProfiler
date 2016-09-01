'''<b>MeasureRadialEntropy</b> measures the variability of an image's
intensity inside a certain object
<hr>
<p>MeasureRadialEntropy divides an object into pie-shaped wedges and
 measures either the mean or median intensity of each.  Once the intensity
 of each wedge has been calculated, the entropy of the bin totals is calculated.</p>

 <p>This module is under construction</p>

'''


import numpy as np
from scipy import stats



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
            doc="""Select the objects whose radial entropy you want to measure.""" % globals())

        self.input_image_name = cps.ImageNameSubscriber(
            "Select an image to measure", doc="""Select the
            grayscale image you want to measure the entropy of.""" % globals())

        self.bin_number=cps.Integer(
            "Input number of bins", 6, minval=3, maxval=60,
            doc="""Number of radial bins to divide your object into.  The minimum number
            of bins allowed is 3, the maximum number is 60.""" % globals())

        self.intensity_measurement=cps.Choice(
            "Which intensity measurement should be used?", ['Mean','Median'], value='Mean',doc="""
            Whether each wedge's mean or median intensity should be used to calculate the entropy.""" % globals())

    def settings(self):
        return [self.input_image_name, self.input_object_name,
                self.intensity_measurement, self.bin_number]


    def run(self, workspace):
        #Import the workspace and the current measurements
        meas = workspace.measurements
        assert isinstance(meas, cpmeas.Measurements)

        statistics = [["Entropy"]]

        workspace.display_data.statistics = statistics

        #Import the settings
        input_image_name = self.input_image_name.value
        input_object_name = self.input_object_name.value
        metric = self.intensity_measurement.value
        bins = self.bin_number.value

        image_set = workspace.image_set

        input_image = image_set.get_image(input_image_name,
                                          must_be_grayscale=True)
        #Read out the pixel data
        pixels = input_image.pixel_data

        object_set = workspace.object_set
        assert isinstance(object_set, cpo.ObjectSet)

        objects = object_set.get_objects(input_object_name)
        labels = objects.segmented

        indexes = objects.indices
        #Calculate the center of the objects- I'm guessing there's a better way to do this but this was already here
        centers, radius = minimum_enclosing_circle(labels, indexes)

        feature = self.get_measurement_name(input_image_name,metric,bins)
        #Do the actual calculation
        entropy=self.slice_and_measure_intensity(pixels,labels,indexes,centers,metric,bins)
        #Add the measurement back into the workspace
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


    def slice_and_measure_intensity(self, pixels, labels, indexes,centers,metric,nbins):
        '''For each object, iterate over the pixels that make up the object, assign them to a bin,
        then call calculate_entropy and return it to run.  I'm sure whatever I did here will make
        Allen and Claire weep tears of bad-code-sadness'''
        entropylist=[]
        for eachindex in range(len(indexes)):
            objects = np.zeros_like(pixels)
            objects[objects==0]=-1
            objects[labels==indexes[eachindex]]= pixels[labels==indexes[eachindex]]
            pixeldict={}
            objectiter=np.nditer(objects,flags=['multi_index'])
            while not objectiter.finished:
                if objectiter[0]!= -1:
                    i1,i2=objectiter.multi_index
                    #Normalize the x,y coordinates to zero
                    center_y,center_x = centers[eachindex]
                    #Do the actual bin calculation
                    sliceno = np.int32((np.pi + np.arctan2(i1-center_y, i2-center_x)) * (nbins / (2 * np.pi)))
                    if sliceno not in pixeldict.keys():
                        pixeldict[sliceno]=[objects[i1,i2]]
                    else:
                        pixeldict[sliceno] += [objects[i1, i2]]
                objectiter.iternext()
            entropy=self.calculate_entropy(pixeldict,metric)
            entropylist.append(entropy)
        entropyarray=np.array(entropylist)
        return entropyarray

    def calculate_entropy(self,pixeldict,metric):
        '''Calculates either the mean or median intensity of each bin as per the user's request,
        normalizes the sum of all of the means/medians to 1, then calculates the entropy'''
        slicemeasurements=[]
        for eachslice in pixeldict.keys():
            if metric=='Mean':
                slicemeasurements.append(np.mean(pixeldict[eachslice]))
            else:
                slicemeasurements.append(np.median(pixeldict[eachslice]))
        slicemeasurements=np.array(slicemeasurements,dtype=float)
        slicemeasurements=slicemeasurements/sum(slicemeasurements)
        entropy=stats.entropy(slicemeasurements)
        return entropy


    def get_feature_name(self,input_image_name,metric,bins):
        '''Return a measurement feature name '''
        return "%s_%s_%d" % (input_image_name, metric, bins)

    def get_measurement_name(self, input_image_name, metric, bins):
        '''Return the whole measurement name'''
        return '_'.join([ENTROPY,
                         self.get_feature_name(input_image_name,metric,bins)])


    def get_measurement_columns(self, pipeline):
        '''Return the column definitions for measurements made by this module'''
        input_object_name = self.input_object_name.value

        input_image_name=self.input_image_name.value
        metric = self.intensity_measurement.value
        bins = self.bin_number.value

        return [(input_object_name,
                 self.get_measurement_name(input_image_name,metric,bins),
                 cpmeas.COLTYPE_FLOAT)]


    def get_categories(self, pipeline, object_name):
        """Get the categories of measurements supplied for the given object name

                pipeline - pipeline being run
                object_name - name of labels in question (or 'Images')
                returns a list of category names
                """
        if object_name == self.input_object_name:
            return [ENTROPY]
        else:
            return []


    def get_measurements(self, pipeline, object_name, category):
        """Get the measurements made on the given object in the given category"""
        if (object_name == self.input_object_name and
                    category == ENTROPY):
            return ["Entropy"]
        else:
            return []



