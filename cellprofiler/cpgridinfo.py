'''cpgridinfo - define a grid structure

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''


import numpy as np

class CPGridInfo(object):
    '''Represents all the parameters of a grid'''
    def __init__(self):
        self.x_location_of_lowest_x_spot = None
        self.y_location_of_lowest_y_spot = None
        self.x_spacing = None
        self.y_spacing = None
        self.rows = None
        self.columns = None
        self.vert_lines_x = None
        self.vert_lines_y = None
        self.horiz_lines_x = None
        self.horiz_lines_y = None
        self.spot_table = None
        self.total_height = None
        self.total_width = None
        self.y_locations = None
        self.x_locations = None
        self.left_to_right = None
        self.top_to_bottom = None
        self.image_width = None
        self.image_height = None

    def serialize(self):
        return dict((k, v) for k, v in self.__dict__.items() if not k.startswith('_'))

    def deserialize(self, serialized_info):
        self.__dict__.update(serialized_info)
