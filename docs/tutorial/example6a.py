'''<b>Example6</b> CellProfiler lifecycle
<hr>
'''

import numpy as np
import scipy.ndimage

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps

class Example6a(cpm.CPModule):
    module_name = "Example6a"
    variable_revision_number = 1
    category = "Image Processing"
    
    def create_settings(self):
        self.input_image_name = cps.ImageNameSubscriber("Input image", "None")
        self.output_image_name = cps.ImageNameProvider("Output image", "Projection")
        self.scale = cps.Float(
            "Scale", .125, .0001, 1,
            doc = """Scale the image dimensions by this fraction""")
        
    def settings(self):
        return [self.input_image_name, self.output_image_name, self.scale]
        
    def prepare_group(self, workspace, grouping, image_numbers):
        '''Prepare to execute a group's cycles
        
        workspace - at this point, the module, pipeline and measurements
                    are valid, but there is no image or object set
                    
        grouping - This is a key/value dictionary that has the metadata used
                   to identify the members of the group.
    
        image_numbers - these are the image numbers for each cycle in the
                        group.
        '''
        d = self.get_dictionary(workspace.image_set_list)
        #
        # Initialize the state here using e6_state_init
        #
        e6_state_init(d, image_numbers)
        
    def run(self, workspace):
        image_number = workspace.measurements.image_number
        d = self.get_dictionary(workspace.image_set_list)
        #
        # Fetch the input image here. 
        #
        image = workspace.image_set.get_image(self.input_image_name.value,
                                              must_be_grayscale=True)
        #
        # call e6_state_append with the image and the dictionary
        #
        e6_state_append(d, image.pixel_data, image_number,
                        self.scale.value)
        #
        # Make an output image. It's always a little tough to know
        # what to do with the output image in aggregation modules. There
        # is a complex way to put a lazily-evaluated image into the image set
        # so it's only calculated on the off-chance it's used, but hard
        # to implement and explain.
        #
        # But you should put *something* in for the output image, but make
        # sure you put the right thing in for the last image cycle.
        #
        # If it's the last image in the group, call e6_state_median to get it
        #
        if image_number == d[K_IMAGE_NUMBERS][-1]:
            output_pixel_data = e6_state_median(d)
        else:
            output_pixel_data = image.pixel_data
        output_image = cpi.Image(output_pixel_data, parent_image = image)
        workspace.image_set.add(self.output_image_name.value, output_image)
        ###
        #
        # Module state can only be stored in the module dictionary and the objects
        # to be stored have to be basic types, lists, tuples, dictionaries or numpy
        # arrays.
        #
        # Normally, you'd wrap all of this inside a class object, but that object
        # doesn't meet the above criteria. So instead, we have a dictionary
        # which we use to accumulate the state
        #
        ###

    def is_aggregation_module(self):
        #
        # Defining "is_aggregation_module" tells the new multiprocessing code
        # to run all cycles for a group in the same worker - different workers
        # will run different groups.
        #
        # This lets each worker accumulate state.
        return True
    
'''Dictionary key to hold the # of cycles in the group'''
K_CYCLE_COUNT = "CycleCount"

'''Dictionary key to hold the group's image numbers'''
K_IMAGE_NUMBERS = "ImageNumbers"

'''Accumulate the array here'''
K_ARRAY = "Array"

'''This is the original image shape'''
K_SHAPE = "Shape"

def e6_state_init(d, image_numbers):
    '''Initialize the module dictionary 
    
    image_numbers - the image numbers for this group
    '''
    d.clear()
    d[K_IMAGE_NUMBERS] = list(image_numbers)
    d[K_CYCLE_COUNT] = len(image_numbers)
    
def e6_state_append(d, image, image_number, scale):
    '''Add an image to the module state
    
    image - an NxM array of grayscale pixel values
    
    image_number - the image number of the current cycle
    
    scale - shrink the image by this scale so we can hold the whole stack.
    '''
    #
    # These are the mapping coordinates to decimate the image.
    #
    # np.newaxis as an index means "duplicate the array contents along this
    # axis". So
    # 
    # np.arange(3)[:, np.newaxis] + np.arange(4)[np.newaxis, :] =
    #     array([[0, 1, 2, 3],
    #            [1, 2, 3, 4],
    #            [2, 3, 4, 5]])
    width = int(image.shape[0] * scale)
    height = int(image.shape[1] * scale)
    sample_i = np.linspace(0, image.shape[0]-1, width)
    sample_j = np.linspace(0, image.shape[1]-1, height)
    k = np.identity(2)
    mapping = \
        sample_i[np.newaxis, :, np.newaxis] * k[0, :, np.newaxis, np.newaxis] +\
        sample_j[np.newaxis, np.newaxis, :] * k[1, :, np.newaxis, np.newaxis]
    
    if not d.has_key(K_ARRAY):
        d[K_ARRAY] = np.zeros((width, height, d[K_CYCLE_COUNT]))
        d[K_SHAPE] = tuple(image.shape)
    a = d[K_ARRAY]
    idx = d[K_IMAGE_NUMBERS].index(image_number)
    mini_image = scipy.ndimage.map_coordinates(image, mapping, mode='reflect')
    a[:, :, idx] = mini_image
    
def e6_state_median(d):
    '''Return a median projection scaled up to the original shape.'''
    a = d[K_ARRAY]
    #
    # Take the median along the stack axis
    #
    mini_image = np.median(a, 2)
    
    width, height = d[K_SHAPE]
    sample_i = np.linspace(0, a.shape[0]-1, width)
    sample_j = np.linspace(0, a.shape[1]-1, height)
    k = np.identity(2)
    mapping = \
        sample_i[np.newaxis, :, np.newaxis] * k[0, :, np.newaxis, np.newaxis] +\
        sample_j[np.newaxis, np.newaxis, :] * k[1, :, np.newaxis, np.newaxis]
    return scipy.ndimage.map_coordinates(mini_image, mapping, mode='reflect')
    