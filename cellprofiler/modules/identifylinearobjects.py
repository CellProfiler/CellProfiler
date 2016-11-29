'''<b>IdentifyLinearObjects</b> identifies linear objects, allowing them to overlap.

<p>This module is under construction.</p>

'''

import numpy as np
from centrosome.cpmorphology import all_connected_components
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from centrosome.cpmorphology import get_line_pts
from scipy.ndimage import binary_erosion, binary_fill_holes
from scipy.ndimage import mean as mean_of_labels

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.preferences as cpprefs
import cellprofiler.settings as cps
import identify as I
from cellprofiler.settings import YES, NO

from skimage.filter.rank import maximum
from skimage.transform import rotate
#from skimage.morphology import binary_dilation

C_LINEAROBJECTS = "LinearObject"
F_ANGLE = "Angle"
M_ANGLE = "_".join((C_LINEAROBJECTS, F_ANGLE))

'''Alpha value when drawing the binary mask'''
MASK_ALPHA = .1
'''Alpha value for labels'''
LABEL_ALPHA = 1.0
'''Alpha value for the shapes'''
LINEAROBJECT_ALPHA = .25

class IdentifyLinearObjects(cpm.CPModule):
    module_name = "IdentifyLinearObjects"
    variable_revision_number = 1
    category = ["Other", "Object Processing"]

    def create_settings(self):
        """Create the settings for the module

        Create the settings for the module during initialization.
        """
        self.image_name = cps.ImageNameSubscriber(
            "Select the input image", cps.NONE,doc="""
            The name of a binary image from a previous module.
            <b>IdentifyLinearObjects</b> will use this image to establish the
            foreground and background for the fitting operation. You can use
            <b>ApplyThreshold</b> to threshold a grayscale image and
            create the binary mask. You can also use a module such as
            <b>IdentifyPrimaryObjects</b> to label each linear object and then use
            <b>ConvertObjectsToImage</b> to make the result a mask.""")

        self.object_name = cps.ObjectNameProvider(
            "Name the linear objects to be identified", "LinearObjects",doc="""
            This is the name for the linear objects. You can refer
            to this name in subsequent modules such as
            <b>IdentifySecondaryObjects</b>""")

        self.object_width = cps.Integer(
            "Linear object width", 10, minval = 1,doc = """
            This is the width (the short axis), measured in pixels,
            of the diamond used as a template when
            matching against the linear object. It should be less than the width
            of a linear object.""")

        self.object_length = cps.Integer(
            "Linear object length", 100, minval= 1,doc = """
            This is the length (the long axis), measured in pixels,
            of the diamond used as a template when matching against the
            linear object. It should be less than the length of a linear object""")

        self.angle_count = cps.Integer(
            "Number of angles", 32, minval = 1,doc = """
            This is the number of different angles at which the
            template will be tried. For instance, if there are 12 angles,
            the template will be rotated by 0&deg;, 15&deg;, 30&deg;, 45&deg; ... 165&deg;.
            The shape is bilaterally symmetric; that is, you will get the same shape
            after rotating it by 180&deg;.""")

        self.wants_automatic_distance = cps.Binary(
            "Automatically calculate distance parameters?", True,doc = """
            This setting determines whether or not
            <b>IdentifyLinearObjects</b> automatically calculates the parameters
            used to determine whether two found-linear object centers belong to the
            same linear object.
            <p>Select <i>%(YES)s</i> to have <b>IdentifyLinearObjects</b>
            automatically calculate the distance from the linear object length
            and width. Select <i>%(NO)s</i> to set the distances manually.</p>"""%globals())

        self.space_distance = cps.Float(
            "Spatial distance", 5, minval = 1,doc = """
            <i>(Used only if not automatically calculating distance parameters)</i><br>
            Enter the distance for calculating the linear object centers, in units of pixels.
            The linear object centers must be at least many pixels apart for the centers to
            be considered two separate linear objects.""")

        self.angular_distance = cps.Float(
            "Angular distance", 30, minval = 1,doc = """
            <i>(Used only if automatically calculating distance parameters)</i><br>
            <b>IdentifyLinearObjects</b> calculates the linear object centers at different
            angles. Two linear object centers are considered to represent different
            linear objects if their angular distance is larger than this number. The
            number is measured in degrees.""")
        
        self.overlap_within_angle = cps.Binary(
            "Combine if overlapping and within angular distance?", False,doc = """
            This setting determines whether or not
            <b>IdentifyLinearObjects</b> merges putative linear objects that are within the
            angular distance specified AND in which pixels from the two 
            linear objects overlap.
            <p>Select <i>%(YES)s</i> to have <b>IdentifyLinearObjects</b>
            merge these linear objects. Select <i>%(NO)s</i> use only the spatial
            distance and angular distance parameters above.</p>"""%globals())

    def settings(self):
        '''The settings as they appear in the pipeline file'''
        return [self.image_name, self.object_name, self.object_width,
                self.object_length, self.angle_count, self.wants_automatic_distance,
                self.space_distance, self.angular_distance, self.overlap_within_angle]

    def visible_settings(self):
        '''The settings as they appear in the user interface'''
        result = [self.image_name, self.object_name, self.object_width,
                  self.object_length, self.angle_count, self.wants_automatic_distance]
        if not self.wants_automatic_distance:
            result += [self.space_distance, self.angular_distance, self.overlap_within_angle]
        return result

    def run(self, workspace):
        '''Run the algorithm on one image set'''
        #
        # Get the image as a binary image
        #
        image_set = workspace.image_set
        image = image_set.get_image(self.image_name.value,
                                    must_be_binary = True)
        mask = image.pixel_data
        if image.has_mask:
            mask = mask & image.mask
        angle_count = self.angle_count.value
        #
        # We collect the i,j and angle of pairs of points that
        # are 3-d adjacent after erosion.
        #
        # i - the i coordinate of each point found after erosion
        # j - the j coordinate of each point found after erosion
        # a - the angle of the structuring element for each point found
        #
        i = np.zeros(0, int)
        j = np.zeros(0, int)
        a = np.zeros(0, int)

        ig, jg = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        #this_idx = 0
        for angle_number in range(angle_count):
            angle = float(angle_number) * np.pi / float(angle_count)
            strel = self.get_diamond(angle)
            erosion = binary_erosion(mask, strel)
            #
            # Accumulate the count, i, j and angle for all foreground points
            # in the erosion
            #
            this_count = np.sum(erosion)
            i = np.hstack((i, ig[erosion]))
            j = np.hstack((j, jg[erosion]))
            a = np.hstack((a, np.ones(this_count, float) * angle))
        #
        # Find connections based on distances, not adjacency
        #
        first, second = self.find_adjacent_by_distance(i,j,a)
        #
        # Do all connected components.
        #
        if len(first) > 0:
            ij_labels = all_connected_components(first, second) + 1
            nlabels = np.max(ij_labels)
            label_indexes = np.arange(1, nlabels + 1)
            #
            # Compute the measurements
            #
            center_x = fix(mean_of_labels(j, ij_labels, label_indexes))
            center_y = fix(mean_of_labels(i, ij_labels, label_indexes))
            #
            # The angles are wierdly complicated because of the wrap-around.
            # You can imagine some horrible cases, like a circular patch of
            # "linear object" in which all angles are represented or a gentle "U"
            # curve.
            #
            # For now, I'm going to use the following heuristic:
            #
            # Compute two different "angles". The angles of one go
            # from 0 to 180 and the angles of the other go from -90 to 90.
            # Take the variance of these from the mean and
            # choose the representation with the lowest variance.
            #
            # An alternative would be to compute the variance at each possible
            # dividing point. Another alternative would be to actually trace through
            # the connected components - both overkill for such an inconsequential
            # measurement I hope.
            #
            angles = fix(mean_of_labels(a, ij_labels, label_indexes))
            vangles = fix(mean_of_labels((a - angles[ij_labels-1])**2,
                                         ij_labels, label_indexes))
            aa = a.copy()
            aa[a > np.pi / 2] -= np.pi
            aangles = fix(mean_of_labels(aa, ij_labels, label_indexes))
            vaangles = fix(mean_of_labels((aa-aangles[ij_labels-1])**2,
                                          ij_labels, label_indexes))
            aangles[aangles < 0] += np.pi
            angles[vaangles < vangles] = aangles[vaangles < vangles]
        else:
            center_x = np.zeros(0, int)
            center_y = np.zeros(0, int)
            angles = np.zeros(0)
            nlabels = 0
            label_indexes = np.zeros(0, int)
            labels = np.zeros(mask.shape, int)
        
        ifull=[]
        jfull=[]
        ij_labelsfull=[]
        labeldict={}
        
        for label_id in np.unique(label_indexes):
            r = np.array(i)*(ij_labels==label_id)
            r=r[r!=0]
            c = np.array(j)*(ij_labels==label_id)
            c=c[c!=0]            
            rect_strel=self.get_rectangle(angles[label_id-1])
            seedmask = np.zeros_like(mask, int)
            seedmask[r,c]=label_id
            
            reconstructedlinearobject=maximum(seedmask,rect_strel)
            reconstructedlinearobject=reconstructedlinearobject*mask
            if self.overlap_within_angle==False:
                itemp,jtemp=np.where(reconstructedlinearobject==label_id)
                ifull+=list(itemp)
                jfull+=list(jtemp)
                ij_labelsfull+=[label_id]*len(itemp)

            else:
                itemp,jtemp=np.where(reconstructedlinearobject==label_id)
                labeldict[label_id]=zip(itemp,jtemp)

        if self.overlap_within_angle==True:
            angledict={}
            for eachangle in range(len(angles)):
                angledict[eachangle+1]=[angles[eachangle]]
            nmerges=1
            while nmerges!=0:
                nmerges=sum([self.mergeduplicates(firstlabel,secondlabel,labeldict,angledict) for firstlabel in label_indexes for secondlabel in label_indexes if firstlabel!=secondlabel])          
            
            newlabels=labeldict.keys()
            newlabels.sort()
            newangles=angledict.keys()
            newangles.sort()
            angles=[]
            for eachnewlabel in range(len(newlabels)):
                ifull+=[int(eachloc[0]) for eachloc in labeldict[newlabels[eachnewlabel]]]
                jfull+=[int(eachloc[1]) for eachloc in labeldict[newlabels[eachnewlabel]]]
                ij_labelsfull+=[eachnewlabel+1]*len(labeldict[newlabels[eachnewlabel]])
                angles.append(np.mean(angledict[newlabels[eachnewlabel]]))
            angles=np.array(angles)   
            
        ijv = np.zeros([len(ifull),3],dtype=int)
        ijv[:,0]=ifull
        ijv[:,1]=jfull
        ijv[:,2]=ij_labelsfull
        
        
        #
        # Make the objects
        #
        object_set = workspace.object_set
        object_name = self.object_name.value
        assert isinstance(object_set, cpo.ObjectSet)
        objects = cpo.Objects()
        objects.ijv = ijv
        objects.parent_image = image
        object_set.add_objects(objects, object_name)
        if self.show_window:
            workspace.display_data.mask = mask
            workspace.display_data.overlapping_labels = [
                    l for l, idx in objects.get_labels()]
        if self.overlap_within_angle==True:
            center_x = np.bincount(ijv[:, 2], ijv[:, 1])[objects.indices] / objects.areas
            center_y = np.bincount(ijv[:, 2], ijv[:, 0])[objects.indices] / objects.areas
                   
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        m.add_measurement(object_name, I.M_LOCATION_CENTER_X, center_x)
        m.add_measurement(object_name, I.M_LOCATION_CENTER_Y, center_y)
        m.add_measurement(object_name, M_ANGLE, angles * 180 / np.pi)
        m.add_measurement(object_name, I.M_NUMBER_OBJECT_NUMBER, label_indexes)
        m.add_image_measurement(I.FF_COUNT % object_name, nlabels)

    def display(self, workspace, figure):
        '''Show an informative display'''
        #import matplotlib
        import cellprofiler.gui.cpfigure
        cplabels=[]
        figure.set_subplots((1, 1))
        assert isinstance(figure, cellprofiler.gui.cpfigure.CPFigureFrame)
        title = self.object_name.value
        cplabels.append(
            dict(name = self.object_name.value,
                 labels = workspace.display_data.overlapping_labels,
                 mode = cellprofiler.gui.cpfigure.CPLDM_ALPHA))
        mask = workspace.display_data.mask
        if mask.ndim == 2:
            figure.subplot_imshow_grayscale(
                0, 0, mask, title = title, cplabels = cplabels)


    def get_diamond(self, angle):
        '''Get a diamond-shaped structuring element

        angle - angle at which to tilt the diamond

        returns a binary array that can be used as a footprint for
        the erosion
        '''
        linearobject_width = self.object_width.value
        linearobject_length = self.object_length.value
        #
        # The shape:
        #
        #                   + x1,y1
        #
        # x0,y0 +                          + x2, y2
        #
        #                   + x3,y3
        #
        x0 = int(np.sin(angle) * linearobject_length/2)
        x1 = int(np.cos(angle) * linearobject_width/2)
        x2 = - x0
        x3 = - x1
        y2 = int(np.cos(angle) * linearobject_length/2)
        y1 = int(np.sin(angle) * linearobject_width/2)
        y0 = - y2
        y3 = - y1
        xmax = np.max(np.abs([x0, x1, x2, x3]))
        ymax = np.max(np.abs([y0, y1, y2, y3]))
        strel = np.zeros((ymax * 2 + 1,
                          xmax * 2 + 1), bool)
        index, count, i, j = get_line_pts(np.array([y0, y1, y2, y3]) + ymax,
                                          np.array([x0, x1, x2, x3]) + xmax,
                                          np.array([y1, y2, y3, y0]) + ymax,
                                          np.array([x1, x2, x3, x0]) + xmax)
        strel[i,j] = True
        strel = binary_fill_holes(strel)
        return strel
        
    def get_rectangle(self, angle):
        linearobject_width = self.object_width.value
        linearobject_length = self.object_length.value
        rect_strel=np.ones([linearobject_length,linearobject_width])
        rect_strel=rotate(rect_strel,-1*angle*180/np.pi,resize=True)
        return rect_strel

    def mergeduplicates(self, firstlabel,secondlabel,labeldict,angledict):
        valtoreturn=0
        if firstlabel in labeldict.keys():
            if secondlabel in labeldict.keys():
                if len(set(labeldict[firstlabel])&set(labeldict[secondlabel])) > 0:
                    anglediff=abs(np.mean(angledict[firstlabel])-np.mean(angledict[secondlabel]))*180/ np.pi
                    if anglediff <= self.angular_distance.value:
                        #nmerges+=1
                        labeldict[firstlabel]=list(set(labeldict[firstlabel]) | set(labeldict[secondlabel]))
                        labeldict.pop(secondlabel)
                        angledict[firstlabel]+=angledict[secondlabel]
                        angledict.pop(secondlabel)
                        valtoreturn=1
        return valtoreturn


    @staticmethod
    def find_adjacent(img1, offset1, count1, img2, offset2, count2, first, second):
        '''Find adjacent pairs of points between two masks

        img1, img2 - binary images to be 8-connected
        offset1 - number the foreground points in img1 starting at this offset
        count1 - number of foreground points in img1
        offset2 - number the foreground points in img2 starting at this offset
        count2 - number of foreground points in img2
        first, second - prior collection of points

        returns augmented collection of points
        '''
        numbering1 = np.zeros(img1.shape, int)
        numbering1[img1] = np.arange(count1) + offset1
        numbering2 = np.zeros(img1.shape, int)
        numbering2[img2] = np.arange(count2) + offset2

        f = np.zeros(0, int)
        s = np.zeros(0, int)
        #
        # Do all 9
        #
        for oi in (-1, 0, 1):
            for oj in (-1, 0, 1):
                f1, s1 = IdentifyLinearObjects.find_adjacent_one(
                    img1, numbering1, img2, numbering2, oi, oj)
                f = np.hstack((f, f1))
                s = np.hstack((s, s1))
        return np.hstack((first, f)), np.hstack((second, s))

    @staticmethod
    def find_adjacent_same(img, offset, count, first, second):
        '''Find adjacent pairs of points in the same mask
        img - binary image to be 8-connected
        offset - where to start numbering
        count - number of foreground points in image
        first, second - prior collection of points

        returns augmented collection of points
        '''
        numbering = np.zeros(img.shape, int)
        numbering[img] = np.arange(count) + offset
        f = np.zeros(0, int)
        s = np.zeros(0, int)
        for oi in (0, 1):
            for oj in (0, 1):
                f1, s1 = IdentifyLinearObjects.find_adjacent_one(
                    img, numbering, img, numbering, oi, oj)
                f = np.hstack((f, f1))
                s = np.hstack((s, s1))
        return np.hstack((first, f)), np.hstack((second, s))

    @staticmethod
    def find_adjacent_one(img1, numbering1, img2, numbering2, oi, oj):
        '''Find correlated pairs of foreground points at given offsets

        img1, img2 - binary images to be correlated
        numbering1, numbering2 - indexes to be returned for pairs
        oi, oj - offset for second image

        returns two vectors: index in first and index in second
        '''
        i1, i2 = IdentifyLinearObjects.get_slices(oi)
        j1, j2 = IdentifyLinearObjects.get_slices(oj)
        match = img1[i1, j1] & img2[i2, j2]
        return numbering1[i1, j1][match], numbering2[i2, j2][match]

    def find_adjacent_by_distance(self, i, j, a):
        '''Return pairs of linear object centers that are deemed adjacent by distance

        i - i-centers of linear objects
        j - j-centers of linear objects
        a - angular orientation of linear objects

        Returns two vectors giving the indices of the first and second
        centers that are connected.
        '''
        if len(i) < 2:
            return (np.zeros(len(i), int), np.zeros(len(i), int))
        if self.wants_automatic_distance:
            space_distance = self.object_width.value
            angle_distance = np.arctan2(self.object_width.value,
                                        self.object_length.value)
            angle_distance += np.pi / self.angle_count.value
        else:
            space_distance = self.space_distance.value
            angle_distance = self.angular_distance.value * np.pi / 180
        #
        # Sort by i and break the sorted vector into chunks where
        # consecutive locations are separated by more than space_distance
        #
        order = np.lexsort((a,j,i))
        i = i[order]
        j = j[order]
        a = a[order]
        breakpoint = np.hstack(([False],i[1:] - i[:-1] > space_distance))
        if np.all(~ breakpoint):
            # No easy win - cross all with all
            first, second = np.mgrid[0:len(i),0:len(i)]
        else:
            # The segment that each belongs to
            segment_number = np.cumsum(breakpoint)
            # The number of elements in each segment
            member_count = np.bincount(segment_number)
            # The index of the first element in the segment
            member_idx = np.hstack(([0], np.cumsum(member_count[:-1])))
            # The index of the first element, for every element in the segment
            segment_start = member_idx[segment_number]
            #
            # Develop the cross-products for each segment. Each segment has
            # member_count * member_count crosses.
            #
            # # of (first,second) pairs in each segment
            cross_size = member_count **2
            # Index in final array of first element of each segment
            segment_idx = np.cumsum(cross_size)
            # relative location of first "first"
            first_start_idx = np.cumsum(member_count[segment_number[:-1]])
            first = np.zeros(segment_idx[-1], int)
            first[first_start_idx] = 1
            # The "firsts" array
            first = np.cumsum(first)
            first_start_idx = np.hstack(([0], first_start_idx))
            second = (np.arange(len(first)) -
                      first_start_idx[first] + segment_start[first])
        mask = ((np.abs((i[first] - i[second]) ** 2 +
                        (j[first] - j[second]) ** 2) <= space_distance ** 2) &
                ((np.abs(a[first] - a[second]) <= angle_distance) |
                 (a[first] + np.pi - a[second] <= angle_distance) |
                 (a[second] + np.pi - a[first] <= angle_distance)))
        return order[first[mask]], order[second[mask]]

    @staticmethod
    def get_slices(offset):
        '''Get slices to use for a pair of arrays, given an offset

        offset - offset to be applied to the second array

        An offset imposes border conditions on an array, for instance,
        an offset of 1 means that the first array has a slice of :-1
        and the second has a slice of 1:. Return the slice to use
        for the first and second arrays.
        '''
        if offset > 0:
            s0, s1= slice(0,-offset), slice(offset, np.iinfo(int).max)
        elif offset < 0:
            s1, s0 = IdentifyLinearObjects.get_slices(-offset)
        else:
            s0 = s1 = slice(0, np.iinfo(int).max)
        return s0, s1

    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements made by this module'''
        object_name = self.object_name.value
        return [(object_name, I.M_LOCATION_CENTER_X, cpmeas.COLTYPE_INTEGER),
                (object_name, I.M_LOCATION_CENTER_Y, cpmeas.COLTYPE_INTEGER),
                (object_name, M_ANGLE, cpmeas.COLTYPE_FLOAT),
                (object_name, I.M_NUMBER_OBJECT_NUMBER, cpmeas.COLTYPE_INTEGER),
                (cpmeas.IMAGE, I.FF_COUNT % object_name, cpmeas.COLTYPE_INTEGER)]

    def get_categories(self, pipeline, object_name):
        if object_name == cpmeas.IMAGE:
            return [ I.C_COUNT ]
        elif object_name == self.object_name:
            return [I.C_LOCATION, I.C_NUMBER, C_LINEAROBJECTS]
        else:
            return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == cpmeas.IMAGE and category == I.C_COUNT:
            return [self.object_name.value]
        elif object_name == self.object_name:
            if category == I.C_LOCATION:
                return [I.FTR_CENTER_X, I.FTR_CENTER_Y]
            elif category == I.C_NUMBER:
                return [I.FTR_OBJECT_NUMBER]
            elif category == C_LINEAROBJECTS:
                return [F_ANGLE]
        return []

