"""project.py - the front-end to a CellProfiler project

This module models a CellProfiler project

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
Copyright (c) 2011 Institut Curie
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
from __future__ import with_statement


import threading

def open_project(path, backend = None):
    '''Open a project file by name'''
    
    return Project(backend(path))

class Project(object):
    '''Project is the grab-bag for CellProfiler experiment-wide information
    
    The Project class is the top-level of the hierarchy of experiment
    information. It contains several types of information:
    
    * The URLs of the 2D image planes in the experiment. Typically, there is one
      URL per image file. However, for composites such as movies, flex files
      or multichannel .TIF files, one URL per plane is saved and enough
      information is encoded in the URL to find the plane within the file.
      It's up to the system at large to use a special URL schema for this,
      like file+image://
      
    * Metadata associated with the URL. This is completely denormalized so that
      a metadata key/value pair is assigned to a single image plane. This means
      that, for cases like a movie with 3 channels of a site in a well,
      well-specific metadata will be applied to every frame of every channel
      of every site in the well, leading to thousand-fold duplication. So be it.
      
    * Urlsets. A urleset is a grouping of image planes created by a user
      so that the user can perform operations on a subset of all URLS. For
      instance, a user might want to extract file name metadata using a
      pattern that applies to only files in a certain directory or wants
      to create an imageset using the files for just one plate.
      
    * Imagesets. An imageset (as opposed to image set in the old terminology)
      is a rectangle of urls with each row representing the images for
      one pipeline iteration and each column representing the images for one
      channel. Generally, but not necessarily, the imageset is organized
      by metadata keys. One metadata key is used to determine a image plane's
      channel and a set of other keys are used to locate the row in which
      the image plane belongs. Each row has a unique set of values for the
      keys.
      
    * Pipelines.
    
    * Measurements.
    
    The project uses some backend to implement a standardized interface.
    The project also synchronizes access to all state somewhat brutally
    using a lock that's taken on entry into any of its functions. If you
    need coarser-grained synchronization to guarantee consistent state at
    some higher level (and I think the operations are coarse enough and
    atomic enough for that not to be necesary), you can take self.lock
    yourself.
      
    '''
    def __init__(self, backend):
        '''Create a project, using a project backend'''
        self.backend = backend
        self.lock = threading.RLock()
        
    def close(self):
        '''Close the project file'''
        with self.lock:
            if self.backend is not None:
                self.backend.rollback()
                self.backend.close()
                self.backend = None
            
    def commit(self):
        '''Commit any changes'''
        with self.lock:
            self.backend.commit()
        
    def rollback(self):
        with self.lock:
            self.backend.rollback()
            
    def __del__(self):
        self.close()
        
    def add_url(self, url):
        '''Add a URL to the list of image files
        
        url - url of an image file
        
        returns the image_id for the URL
        '''
        with self.lock:
            return self.backend.add_url(url)
        
    def get_url_image_id(self, url):
        '''Get the image ID for a url'''
        with self.lock:
            return self.backend.get_url_image_id(url)
    
    def get_url(self, image_id):
        '''Get a URL given its image id
        
        image_id - the image ID for a url, can be a sequence
                   in which case, we return a sequence of urls
        '''
        with self.lock:
            return self.backend.get_url(image_id)
    
    def remove_url_by_id(self, image_id):
        '''Remove a URL, using its image id'''
        with self.lock:
            self.backend.remove_url_by_id(image_id)
    
    def remove_url(self, url):
        '''Remove a url by name'''
        with self.lock:
            image_id = self.get_url_image_id(url)
            if image_id is not None:
                self.remove_url_by_id(image_id)
                
    def add_directory(self, name, parent=None):
        '''Add a directory to the project
        
        Put a directory into the directory table, optionally linking it
        to its parent.
        
        name - the directory's URL
        
        parent - the name of the parent directory.
        '''
        with self.lock:
            self.backend.add_directory(name, parent)
            
    def get_directories(self):
        '''Return all directories in the project
        '''
        with self.lock:
            return self.backend.get_directories()
            
    def get_root_directories(self):
        '''Return all root directories in the project
        '''
        with self.lock:
            return self.backend.get_root_directories()
        
    def get_subdirectories(self, parent):
        '''Return all immediate subdirectories of the parent
        
        parent - the name of the parent directory
        '''
        with self.lock:
            return self.backend.get_subdirectories(parent)
        
    def remove_directory(self, name):
        '''Remove a directory and its subdirectories
        
        name - name of the directory to remove
        
        Note: does not remove the URLs "in" the directory.
        '''
        with self.lock:
            self.backend.remove_directory(name)
            
    def get_metadata_keys(self):
        '''Return all the metadata keys for this project
        
        returns a sequence of key names
        '''
        with self.lock:
            return self.backend.get_metadata_keys()
        
    def get_metadata_values(self, key):
        '''Return the metadata values that have been assigned to a particular key
        
        key - a metadata key.
        '''
        with self.lock:
            return self.backend.get_metadata_values(key)
    
    def add_image_metadata(self, keys, values, image_id):
        '''Assign metadata values to image planes
        
        keys - a sequence of the metadata keys to be assigned
        
        values - either a sequence of values for the keys or, if being assigned
                to multiple frames, an N x M array of values where N is
                the number of frames and M is the number of keys
                
        image_id - the image_id of the image or a 1-d array of image_ids
        '''
        with self.lock:
            self.backend.add_image_metadata(keys, values, image_id)
        
    def get_image_metadata(self, image_id):
        '''Get all metadata key/value pairs for an image
        
        image_id - the image's image_id
        
        returns a dictionary of metadata key/value pairs for the image
        '''
        with self.lock:
            return self.backend.get_image_metadata(image_id)
        
    def remove_image_metadata(self, key, image_id):
        '''Remove metadata values from image planes
        
        key - a key to be removed from a single image's metadata or from
              a sequence of images
         
        image_id - the image_id of the image or a 1-d array of image ids
        '''
        with self.lock:
            self.backend.remove_image_metadata(key, image_id)

    def get_images_by_metadata(self, keys, values=None, urlset=None):
        '''Return images by metadata key and value
        
        keys - the metadata keys for the operation
        values - if None, return images, ordered by value.
                 if a single tuple of values, return images matching
                 those values for the keys.
        urlset - if None, match any images, if not None, match only
                 images in the urlset
        
        returns an array where each row represents a frame. For M keys,
        the first M values of the array are the metadata values for
        each of the keys for that row's frame. The last value in each
        row of the array is the image_id of the image.
        '''
        with self.lock:
            return self.backend.get_images_by_metadata(keys, values, urlset)
    
    def make_urlset(self, name):
        '''Create a frames with a given name
        
        A urlset is a collection of image URLs.
        Sometimes, you might want to only run a pipeline on some
        of the images in the dataset (for instance, illumination correction
        images or one plate of images). So image set operations take
        urlsets to allow this sort of flexibility.
        
        name - the name of the urlset
        '''
        with self.lock:
            return self.backend.make_urlset(name)
        
    def get_urlset_names(self):
        '''Return the names of all of the urlsets'''
        with self.lock:
            return self.backend.get_urlset_names()
    
    def remove_urlset(self, name):
        '''Delete a urlset
        
        name - the name of the urlset to delete
        '''
        with self.lock:
            return self.backend.remove_urlset(name)
    
    def add_images_to_urlset(self, name, image_ids):
        '''Add images to a urlset
        
        name - the name of the urlset
        
        image_ids - a 1-d array of image ids to add to the urlset
        '''
        with self.lock:
            self.backend.add_images_to_urlset(name, image_ids)
        
    def remove_images_from_urlset(self, name, image_ids):
        '''Remove frames from a urlset
        
        name - the name of the urlset

        image_ids - a 1-d array of image ids to add to the urlset
        '''
        with self.lock:
            self.backend.remove_images_from_urlset(name, image_ids)
        
    def get_urlset_members(self, name):
        '''Return all images in the urlset
        
        Returns a 1-d array of the image ids in the urlset
        '''
        with self.lock:
            return self.backend.get_urlset_members(name)
    
    def create_imageset(self, name, keys, channel_key, 
                        channel_values = None, urlset=None):
        '''Create an image set
        
        name - the name of the image set
        
        urlset - the name of the urlset. If None, operate on all frames
        
        keys - the metadata keys that uniquely define an image set row
        
        channel_key - the metadata key that assigns a frame to a channel
        
        channel_values - the channels to add to the image set. For instance,
        if the channel_key is "wavelength" and you only want "w1" and "w2"
        in the imageset, but not "w3", channel_values would be ["w1", "w2"].
        If None, accept all channel values.
        
        Create an image set where each row in the image set has unique values
        for the set of metadata keys. For instance, the keys might be
        "Plate", "Well" and "Site" and a row might have values "P-12345",
        "A05" and "s3". Each (conceptual) row has columns which are the
        possible values for the channel key in the  urlset. For instance,
        the channel_key might be "Wavelength" with values "w1" and "w2".
        
        The result of the operation is an image set whose rows can be referenced
        by image number or by key values.
        '''
        with self.lock:
            return self.backend.create_imageset(name, keys, channel_key,
                                                channel_values, urlset)
        
    def remove_imageset(self, name):
        '''Delete the named imageset'''
        with self.lock:
            self.backend.remove_imageset(name)
        
    def get_imageset_row_count(self, name):
        '''Return the number of rows in the named imageset'''
        with self.lock:
            return self.backend.get_imageset_row_count(name)
        
    def get_imageset_row_images(self, name, image_number):
        '''Return the images in an imageset row
        
        name - the name of the imageset
        image_number - the one-based image number for the row
        
        returns a dictionary whose key is channel value and whose value
        is a 1-d array of image_ids for that row. An array might be empty for 
        some channel value (missing image) or might contain more than one
        image id (duplicate images).
        '''
        with self.lock:
            return self.backend.get_imageset_row_images(name, image_number)
        
    def get_imageset_row_metadata(self, name, image_number):
        '''Return the imageset row's metadata values
        
        name - the name of the imageset
        
        image_number - the one-based image number for the row
        
        returns a dictionary of metadata key and value
        '''
        with self.lock:
            return self.backend.get_imageset_row_metadata(name, image_number)
        
    def get_problem_imagesets(self, name):
        '''Return imageset rows with missing or duplicate images
        
        name - the name of the imageset
        
        returns a sequence of tuples of image_number, channel name and the
        count of number of images for channels with no or duplicate images.
        '''
        with self.lock:
            return self.backend.get_problem_imagesets(name)
        
    def add_image_to_imageset(self, image_id, name, image_number, channel):
        '''Add a single image to an imageset
        
        This method can be used to patch up an imageset if the user
        wants to manually correct it image by image.
        
        image_id - the image id of the image
        
        name - the name of the imageset
        
        image_number - add the image to the row with this image number
        
        channel - the image's channel
        '''
        with self.lock:
            self.backend.add_image_to_imageset(image_id, name, image_number, channel)
    
    def remove_image_from_imageset(self, name, image_id):
        '''Remove an image frame from an imageset
        
        Remove the image with the given image_id from the imageset.
        '''
        with self.lock:
            self.backend.remove_image_from_imageset(name, image_id)
        
    def add_channel_to_imageset(self, name, keys, 
                                urlset = None,
                                channel_name = None, 
                                channel_key = None, channel_value = None):
        '''Add a channel to an imageset
        
        Add the images in a urlset to create a new channel in an imageset.
        Each image might be applied to one or more imagesets and might be
        applied to none if its metadata values don't match any in the
        imageset.
        
        The key set should be a subset (x[:] is a subset of x[:]) of that of the 
        imageset. The function will identify all rows in the imageset whose
        metadata values match those of an image in the urlset and will include
        that image in each of the matching rows.
        
        For instance, image correction might be done per-plate and the user has
        a urlset consisting of one image per plate. This function would
        be called with keys = ["Plate"] to assign one image to all rows
        for each plate.
        
        name - the name of the imageset
        
        urlset - the name of the urlset containing the images to be applied
        
        keys - the key names of the unique metadata for the frames
        
        channel_name - the name to be assigned to the channel. If none,
                       the channel_value will be the channel name
                       
        channel_key - the channel metadata key. If None, all frames in the
                      urlset will be assigned to the imageset
                      
        channel_value - the metadata value for the channel, for instance,
                        "w1" for "wavelength"
        '''
        with self.lock:
            return self.backend.add_channel_to_imageset(
                name, keys, urlset, channel_name, channel_key, channel_value)
    
