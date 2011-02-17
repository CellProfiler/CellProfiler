'''
'''
import omero
from omero import client_wrapper
from omero.util import script_utils

import Image

from cellprofiler.modules import loadimages as cpli
import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps


class OMEROImageProvider(cpi.AbstractImageProvider):
    def __init__(self, blitzcon, name):
        '''Initializer
        '''
        self.__blitzcon = blitzcon
        self.__name = name

    def provide_image(self, omero_image_id):
        """Load an image from an omero_image_id
        """
        blitzcon = self.get_blitzcon()
        rawPixelsStore = blitzcon.createRawPixelsStore()
        theC, theZ, theT = 0, 0, 0
        bypassOriginalFile = True
        im = blitzcon.getImage(20627)
        pixels = im.getPrimaryPixels()
        # gets a 2d numpy array.
        plane = script_utils.downloadPlane(rawPixelsStore, pixels, theZ, theC, theT)
        return plane

    def get_blitzcon(self):
        if not self.blitzcon.isConnected():
            self.__blitzcon.connect()
        return self.__blitzcon
    
    def get_name(self):
        return self.__name
    

class OMEROLoader(cpm.CPModule):
    module_name = "OMEROLoader"
    category = "File Processing"
    variable_revision_number = 1
    project_ids = {}
    dataset_ids = {}
    __blitzcon = None

    def create_settings(self):
        self.host = cps.Text('Host:', 'ome2-copy.fzk.de')
        self.username = cps.Text('User name:', 'afraser')
        self.password = cps.Text('User name:', 'demo')
        self.project = cps.Choice('Project:', self.get_projects_list(), 
                                  choices_fn=self.get_projects_list)
        self.dataset = cps.Choice('Dataset:', self.get_datasets_list(), 
                                  choices_fn=self.get_datasets_list)
        self.channel = cps.ImageNameProvider('Name the first channel:', 'DNA')
        
    def settings(self):
        return [self.host, self.username, self.password, self.project, 
                self.dataset, self.channel]
    
    def visible_settings(self):
        return self.settings()
    
    def prepare_run(self, pipeline, image_set_list, frame):
        """Prepare the image set list for a run (& whatever else you want to do)
        
        pipeline - the pipeline being run
        image_set_list - add any image sets to the image set list
        frame - parent frame of application if GUI enabled, None if GUI
                disabled
        
        return True if operation completed, False if aborted 
        """
        blitzcon = self.get_blitz_connection()
        dataset = blitzcon.getDataset(self.dataset_ids[self.dataset.value])
        d = self.get_dictionary(image_set_list)
        for i, im in enumerate(dataset.listChildren()):
            pixels = im.getPrimaryPixels()
            d[i] = pixels.getId().getValue()
            
        provider = OMEROImageProvider(blitzcon, self.channel.value)
        image_set.providers.append(provider)
        
        for i in range(len(d)):
            image_set = image_set_list.get_image_set(i)

    def run(self, workspace):
        d = self.get_dictionary()
        img_id = d[workspace.image_set.number]        
        imdata = workspace.get_image_set()
        Image.fromarray(imdata).show()

        #
        # Add measurements
        #
        m.add_image_measurement("_".join((cpli.C_PATH_NAME, cpli.C_FILE_NAME)), filename)
        if len(path) > 0:
            full_path = os.path.join(full_path, path)
        m.add_image_measurement("_".join((path_name_category, name)), full_path)
        digest = hashlib.md5()
        digest.update(np.ascontiguousarray(pixel_data).data)
        m.add_measurement('Image',"_".join((C_MD5_DIGEST, name)), digest.hexdigest())
        m.add_image_measurement("_".join((C_SCALING, name)), image.scale)
        
        
    def is_interactive(self):
        return False
    
    def get_blitz_connection(self):
        '''Returns a connection to the OMERO blitz server omero.gateway._BlitzGateway
        or None if a connection cannot be established with the existing settings.

        Depends on self.username, self.password, self.host
        '''
        if self.__blitzcon is not None and self.__blitzcon.isConnected():
            return self.__blitzcon
        
        self.__blitzcon = client_wrapper(self.username.value,
                                         self.password.value,
                                         host=self.host.value, 
                                         port=4064)
        print 'connecting to OMERO.blitz'
        if not self.__blitzcon.connect():
            return None
        return self.__blitzcon

    def get_projects_list(self, pipeline=None):
        blitzcon = self.get_blitz_connection()
        if blitzcon is None:
            return []
        self.project_ids = {}
        for p in blitzcon.listProjects():
            assert p.getName() not in self.project_ids, 'OMEROLoader detected 2 OMERO projects with the same name, "%s".'%(p.getName())
            self.project_ids[p.getName()] = p.id
        return self.project_ids.keys()
    
    def get_datasets_list(self, pipeline=None):
        blitzcon = self.get_blitz_connection()
        project_id = self.project_ids.get(self.project.value, None)
        if project_id is None:
            return []
        project = blitzcon.getProject(project_id)
        self.dataset_ids = {}
        for d in project.listChildren():
            assert d.getName() not in self.dataset_ids, 'OMEROLoader detected 2 OMERO datasets with the same name, "%s".'%(d.getName())
            self.dataset_ids[d.getName()] = d.id
        return self.dataset_ids.keys()
        
    
    #def display(self, workspace):
        #pass