'''<b>Align</b> aligns images relative to each other, for example to correct 
shifts in the optical path of a microscope in each channel of a multi-channel 
set of images.
<hr>

For two or more input images, this module determines the optimal alignment 
among them. Aligning images is useful to obtain proper measurements of the 
intensities in one channel based on objects identified in another channel, 
for example. Alignment is often needed when the microscope is not perfectly 
calibrated. It can also be useful to align images in a time-lapse series of 
images.  The module stores the amount of shift between images as a
measurement, which can be useful for quality control purposes.

Features that can be measured by this module:
<ul> 
<li>Xshift_Image1NamevsImage2Name  (e.g., Xshift_BluevsRed)</li>
<li>Yshift_Image1NamevsImage2Name  (e.g., Yshift_BluevsRed)</li>
<li>Xshift_Image1NamevsImage3Name  (e.g., Xshift_RedvsGreen)</li>
<li>Yshift_Image1NamevsImage3Name  (e.g., Yshift_RedvsGreen)</li>
<li>etc...</li>
</ul>

'''
__version__ = "$Revision$"

import numpy as np
from scipy.fftpack import fft2, ifft2
import scipy.ndimage as scind
import scipy.sparse

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
from cellprofiler.cpmath.filter import stretch

M_MUTUAL_INFORMATION = 'Mutual Information'
M_CROSS_CORRELATION = 'Normalized Cross Correlation'
M_ALL = (M_MUTUAL_INFORMATION, M_CROSS_CORRELATION)

A_SIMILARLY = 'Similarly'
A_SEPARATELY = 'Separately'

MEASUREMENT_FORMAT = "Align_%sshift_%s_vs_%s"

class Align(cpm.CPModule):
    module_name = "Align"
    category = 'Image Processing'
    variable_revision_number = 1

    def create_settings(self):
        self.first_input_image = cps.ImageNameSubscriber("Select the first input image",
                                                         "None",doc="""
                                                         What is the name of the first image to align?  Regardless of the number of input images, they will all be aligned
with respect to the first image.""")
        self.first_output_image = cps.ImageNameProvider("Name the first output image",
                                                        "AlignedRed",doc="""
                                                        What do you want to call the aligned first image?""")
        self.separator_1 = cps.Divider(line=False)
        self.second_input_image = cps.ImageNameSubscriber("Select the second input image",
                                                          "None",doc="""
                                                          What is the name of the second image to align?""")
        self.second_output_image = cps.ImageNameProvider("Name the second output image",
                                                         "AlignedGreen",doc="""
                                                         What do you want to call the aligned second image?""")
        self.separator_2 = cps.Divider(line=False)
        self.additional_images = []
        self.add_button = cps.DoSomething("", "Add another image",
                                          self.add_image)
        self.alignment_method = cps.Choice("Select the alignment method",
                                           M_ALL, doc='''
             Which alignment method would you like to use? Two options are available:<br>
             <ul>
             <li><i>Mutual Information method:</i> With this method, alignment works whether the 
             images are correlated (bright in one = bright in the other) or 
             anti-correlated (bright in one = dim in the other). </li>
             <li><i>Normalized Cross Correlation method:</i> With this method, alignment works only 
             when the images are correlated (bright in one = bright in the 
             other). When using the cross correlation method, the second 
             image should serve as a template and be smaller than the first 
             image selected.</li>
             </ul>''')
        self.wants_cropping = cps.Binary("Crop output images to retain just the aligned regions?",
                                         True, doc='''
             If you choose to crop, all output images are cropped to retain 
             just those regions that exist in all channels after alignment. 
             If you do not choose to crop, the unaligned portions of each
             image are padded (with zeroes) and appear as black space.''')
    
    def add_image(self):
        '''Add an image + associated questions and buttons'''
        group = cps.SettingsGroup()
        group.append("input_image_name", 
                     cps.ImageNameSubscriber("Select the additional image?",
                                            "None",doc="""
                                            What is the name of the additional image to align?"""))
        group.append("output_image_name",
                     cps.ImageNameProvider("Name the output image",
                                            "AlignedBlue",doc="""
                                            What do you want to call the aligned image?"""))
        group.append("align_choice",
                     cps.Choice("Select how the alignment is to be applied",
                                               [A_SIMILARLY, A_SEPARATELY],doc="""
                                               Do you want to align this image similarly to the second one or do you 
                                               want to calculate a separate alignment to the first image?<br>
                                               <ul>
                                               <li><i>Similarly:</i> The same alignment measurements obtained from
                                               the first two input images are applied to this additional image.</li>
                                               <li><i>Separately:</i> A new set of alignment measurements are
                                               calculated for this additional image using the alignment method
                                               specified with respect to the first input image.</li>
                                               </ul>"""))
        group.append("remover", cps.RemoveSettingButton("", "Remove above image", self.additional_images, group))
        group.append("divider", cps.Divider(line=False))
        self.additional_images.append(group)

    def settings(self):
        result = [self.alignment_method, self.wants_cropping]
        
        result += [self.first_input_image, self.first_output_image,
                  self.second_input_image, self.second_output_image]
        for additional in self.additional_images:
            result += [additional.input_image_name, additional.output_image_name, additional.align_choice]
        return result

    def prepare_settings(self, setting_values):
        assert (len(setting_values)-6)% 3 == 0
        n_additional = (len(setting_values)-6)/3
        del self.additional_images[:]
        while len(self.additional_images) < n_additional:
            self.add_image()

    def visible_settings(self):
        result = [self.alignment_method, self.wants_cropping]
        
        result += [self.first_input_image, self.first_output_image, self.separator_1,
                  self.second_input_image, self.second_output_image, self.separator_2]
        for additional in self.additional_images:
            result += additional.unpack_group()
        result += [self.add_button]
        return result

    def run(self, workspace):
        i_min = np.iinfo(int).max
        j_min = np.iinfo(int).max
        images = ([self.first_input_image, self.second_input_image] +
                  [additional.input_image_name 
                   for additional in self.additional_images])
        for image in images:
            img = workspace.image_set.get_image(image.value).pixel_data
            if img.shape[0] < i_min or img.shape[1] < j_min:
                most_cropped_image_name = image.value
                i_min,j_min = img.shape[0:2]

        off_x, off_y = self.align(workspace, self.first_input_image.value,
                                  self.second_input_image.value,
                                  most_cropped_image_name)
        statistics = [[0,self.second_input_image.value, 
                       self.second_output_image.value,
                       -off_x, -off_y]]
        self.apply_alignment(workspace,
                             self.second_input_image.value, 
                             self.second_output_image.value,
                             off_x, off_y, most_cropped_image_name)
        self.crop(workspace, self.first_input_image.value,
                  self.first_output_image.value,
                  most_cropped_image_name)
        for additional in self.additional_images:
            if additional.align_choice == A_SIMILARLY:
                self.apply_alignment(workspace,
                                     additional.input_image_name.value, 
                                     additional.output_image_name.value,
                                     off_x, off_y, most_cropped_image_name)
                a_off_x = off_x
                a_off_y = off_y
            else:
                a_off_x, a_off_y = self.align(workspace, 
                                              self.first_input_image.value,
                                              additional.input_image_name.value,
                                              most_cropped_image_name)
                self.apply_alignment(workspace,
                                     additional.input_image_name.value,
                                     additional.output_image_name.value,
                                     a_off_x, a_off_y, most_cropped_image_name)
            statistics += [[len(statistics),
                            additional.input_image_name.value,
                            additional.output_image_name.value,
                            -a_off_x, -a_off_y]]
        #
        # Write the measurements
        #
        for index, input_name, output_name, t_off_x, t_off_y in statistics:
            for axis, value in (('X',t_off_x),('Y',t_off_y)):
                feature = (MEASUREMENT_FORMAT %
                           (axis, self.first_output_image.value,
                            output_name))
                workspace.measurements.add_image_measurement(feature, value)
            
        # save data for display
        workspace.display_data.statistics = statistics
        workspace.display_data.most_cropped_image_name = most_cropped_image_name
    
    def display(self, workspace):
        '''Display the overlaid images
        
        workspace - the workspace being run
        statistics - a list of lists:
            0: index of this statistic
            1: input image name of image being aligned
            2: output image name of image being aligned
            3: x offset
            4: y offset
        '''
        statistics = workspace.display_data.statistics
        most_cropped_image_name = workspace.display_data.most_cropped_image_name
        image_set = workspace.image_set
        crop_image = image_set.get_image(most_cropped_image_name)
        first_image = image_set.get_image(self.first_input_image.value,
                                          must_be_grayscale=True)
        first_pixels = crop_image.crop_image_similarly(first_image.pixel_data)
        figure = workspace.create_or_find_figure(subplots=(2,len(statistics)))
        for i, input_name, output_name, off_x, off_y in statistics:
            other_image = image_set.get_image(input_name,
                                              must_be_grayscale=True)
            other_pixels =\
                crop_image.crop_image_similarly(other_image.pixel_data)
            img = np.dstack((first_pixels,
                             other_pixels,
                             np.zeros(first_pixels.shape)))
            title = ("Unaligned images: %s and %s"%
                     (self.first_input_image.value, input_name))
            figure.subplot_imshow_color(0,i,img,title)
            
            other_image = image_set.get_image(output_name,
                                              must_be_grayscale=True)
            other_pixels = other_image.pixel_data
            img = np.dstack((first_pixels,
                             other_pixels,
                             np.zeros(first_pixels.shape)))
            title = ("Aligned images: %s and %s\nX offset: %d, Y offset: %d"%
                     (self.first_output_image.value, output_name,
                      off_x, off_y))
            figure.subplot_imshow_color(1,i,img,title)

    def is_interactive(self):
        return False
        
    def align(self, workspace, input1_name, input2_name, most_cropped_image_name):
        '''Align the second image with the first
        
        Calculate the alignment offset that must be added to indexes in the
        first image to arrive at indexes in the second image.
        
        Returns the x,y (not i,j) offsets.
        '''
        image1 = workspace.image_set.get_image(input1_name,
                                               must_be_grayscale=True)
        image1_pixels = image1.pixel_data
        image2 = workspace.image_set.get_image(input2_name,
                                               must_be_grayscale=True)
        image2_pixels = image2.pixel_data
        cropping_image = workspace.image_set.get_image(most_cropped_image_name)
        image1_pixels = cropping_image.crop_image_similarly(image1_pixels)
        image2_pixels = cropping_image.crop_image_similarly(image2_pixels)
        if self.alignment_method == M_CROSS_CORRELATION:
            return self.align_cross_correlation(image1_pixels, image2_pixels)
        else:
            return self.align_mutual_information(image1_pixels, image2_pixels)
    
    def align_cross_correlation(self, pixels1, pixels2):
        '''Align the second image with the first using max cross-correlation
        
        returns the x,y offsets to add to image1's indexes to align it with
        image2
        
        Many of the ideas here are based on the paper, "Fast Normalized
        Cross-Correlation" by J.P. Lewis 
        (http://www.idiom.com/~zilla/Papers/nvisionInterface/nip.html)
        which is frequently cited when addressing this problem.
        '''
        #
        # We double the size of the image to get a field of zeros
        # for the parts of one image that don't overlap the displaced
        # second image.
        #
        assert tuple(pixels1.shape)==tuple(pixels2.shape)
        s = np.array(pixels1.shape)
        fshape = s*2
        #
        # Calculate the # of pixels at a particular point
        #
        i,j = np.mgrid[-pixels1.shape[0]:pixels1.shape[0],
                       -pixels1.shape[1]:pixels1.shape[1]]
        unit = np.abs(i*j).astype(float)
        unit[unit<1]=1 # keeps from dividing by zero in some places
        #
        # Normalize the pixel values around zero which does not affect the
        # correlation, keeps some of the sums of multiplications from
        # losing precision and precomputes t(x-u,y-v) - t_mean
        #
        pixels1 = pixels1-np.mean(pixels1)
        pixels2 = pixels2-np.mean(pixels2)
        #
        # Lewis uses an image, f and a template t. He derives a normalized
        # cross correlation, ncc(u,v) =
        # sum((f(x,y)-f_mean(u,v))*(t(x-u,y-v)-t_mean),x,y) /
        # sqrt(sum((f(x,y)-f_mean(u,v))**2,x,y) * (sum((t(x-u,y-v)-t_mean)**2,x,y)
        #
        # From here, he finds that the numerator term, f_mean(u,v)*(t...) is zero
        # leaving f(x,y)*(t(x-u,y-v)-t_mean) which is a convolution of f
        # by t-t_mean.
        #
        fp1 = fft2(pixels1,fshape)
        fp2 = fft2(pixels2,fshape)
        corr12 = ifft2(fp1 * fp2.conj()).real
        
        #
        # Use the trick of Lewis here - compute the cumulative sums
        # in a fashion that accounts for the parts that are off the
        # edge of the template.
        #
        # We do this in quadrants:
        # q0 q1
        # q2 q3
        # For the first, 
        # q0 is the sum over pixels1[i:,j:] - sum i,j backwards
        # q1 is the sum over pixels1[i:,:j] - sum i backwards, j forwards
        # q2 is the sum over pixels1[:i,j:] - sum i forwards, j backwards
        # q3 is the sum over pixels1[:i,:j] - sum i,j forwards
        #
        # The second is done as above but reflected lr and ud
        #
        p1_sum = np.zeros(fshape)
        p1_sum[:s[0],:s[1]] = cumsum_quadrant(pixels1, False, False)
        p1_sum[:s[0],s[1]:] = cumsum_quadrant(pixels1, False, True)
        p1_sum[s[0]:,:s[1]] = cumsum_quadrant(pixels1, True, False)
        p1_sum[s[0]:,s[1]:] = cumsum_quadrant(pixels1, True, True)
        #
        # Divide the sum over the # of elements summed-over
        #
        p1_mean = p1_sum / unit
        
        p2_sum = np.zeros(fshape)
        p2_sum[:s[0],:s[1]] = cumsum_quadrant(pixels2, False, False)
        p2_sum[:s[0],s[1]:] = cumsum_quadrant(pixels2, False, True)
        p2_sum[s[0]:,:s[1]] = cumsum_quadrant(pixels2, True, False)
        p2_sum[s[0]:,s[1]:] = cumsum_quadrant(pixels2, True, True)
        p2_sum = np.fliplr(np.flipud(p2_sum))
        p2_mean = p2_sum / unit
        #
        # Once we have the means for u,v, we can caluclate the
        # variance-like parts of the equation. We have to multiply
        # the mean^2 by the # of elements being summed-over
        # to account for the mean being summed that many times.
        #
        p1sd = np.sum(pixels1**2) - p1_mean**2 * np.product(s)
        p2sd = np.sum(pixels2**2) - p2_mean**2 * np.product(s)
        #
        # There's always chance of roundoff error for a zero value
        # resulting in a negative sd, so limit the sds here
        #
        sd = np.sqrt(np.maximum(p1sd * p2sd, 0))
        corrnorm = corr12 / sd
        #
        # There's not much information for points where the standard
        # deviation is less than 1/100 of the maximum. We exclude these
        # from consideration.
        # 
        corrnorm[(unit < np.product(s) / 2) &
                 (sd < np.mean(sd) / 100)] = 0
        i,j = np.unravel_index(np.argmax(corrnorm ),fshape)
        #
        # Reflect values that fall into the second half
        #
        if i > pixels1.shape[0]:
            i = i - fshape[0]
        if j > pixels1.shape[1]:
            j = j - fshape[1]
        return j,i
    
    def align_mutual_information(self, pixels1, pixels2):
        '''Align the second image with the first using mutual information
        
        returns the x,y offsets to add to image1's indexes to align it with
        image2
        
        The algorithm computes the mutual information content of the two
        images, offset by one in each direction (including diagonal) and
        then picks the direction in which there is the most mutual information.
        From there, it tries all offsets again and so on until it reaches
        a local maximum.
        '''
        def mutualinf(x,y):
            return entropy(x) + entropy(y) - entropy2(x,y)
        
        best = mutualinf(pixels1, pixels2)
        i = 0
        j = 0
        while True:
            last_i = i
            last_j = j
            for new_i in range(last_i-1,last_i+2):
                for new_j in range(last_j-1, last_j+2):
                    if new_i == 0 and new_j == 0:
                        continue
                    p2, p1 = offset_slice(pixels2,pixels1, new_i, new_j)
                    info = mutualinf(p1,p2)
                    if info > best:
                        best = info
                        i = new_i
                        j = new_j
            if i == last_i and j == last_j:
                return j,i
        
    def apply_alignment(self, workspace, input_image_name, output_image_name,
                        off_x, off_y, most_cropped_image_name):
        image = workspace.image_set.get_image(input_image_name,
                                              must_be_grayscale = True)
        most_cropped_image = workspace.image_set.get_image(most_cropped_image_name)
        
        '''Create an output image that's offset by the given # of pixels'''
        pixels = most_cropped_image.crop_image_similarly(image.pixel_data)
        output_pixels = np.zeros(pixels.shape)
        #
        # Copy the input to the output
        #
        p1,p2 = offset_slice(pixels, output_pixels, off_y, off_x)
        p2[:,:] = p1[:,:]
        if off_x != 0 or off_y != 0:
            #
            # Construct a mask over the zero-filling
            #
            mask = np.zeros(output_pixels.shape, bool)
            p1, m2 = offset_slice(pixels, mask, off_y, off_x)
            m2[:,:] = True
            
            if image.has_mask:
                mask = (mask & most_cropped_image.crop_image_similarly(image.mask))
        elif image.has_mask:
            mask = most_cropped_image.crop_image_similarly(image.mask)
        else:
            mask = None
        output_image = cpi.Image(output_pixels, 
                                 mask = mask, 
                                 crop_mask = most_cropped_image.crop_mask,
                                 parent_image = image)
        workspace.image_set.add(output_image_name, output_image)
    
    def crop(self, workspace, input_image_name, output_image_name,
             most_cropped_image_name):
        '''Crop and save an image'''
        image = workspace.image_set.get_image(input_image_name,
                                              must_be_grayscale = True)
        most_cropped_image = workspace.image_set.get_image(most_cropped_image_name)
        pixels = most_cropped_image.crop_image_similarly(image.pixel_data)
        output_image = cpi.Image(pixels, 
                                 crop_mask = most_cropped_image.crop_mask,
                                 parent_image = image)
        workspace.image_set.add(output_image_name, output_image)
    
    def get_measurement_columns(self, pipeline):
        '''return the offset measurements'''
        
        targets = ([self.second_output_image.value] +
                   [additional.output_image_name.value
                    for additional in self.additional_images])
        columns = []
        for axis in ('X','Y'):
            columns += [(cpmeas.IMAGE, 
                         MEASUREMENT_FORMAT%(axis,self.first_output_image.value,
                                             target),
                         cpmeas.COLTYPE_INTEGER)
                         for target in targets]
        return columns

    def upgrade_settings(self, setting_values, 
                         variable_revision_number, 
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 5:
            #
            # The Matlab align module has the following layout
            # 0:  Image1Name
            # 1:  AlignedImage1Name
            # 2:  Image2Name
            # 3:  AlignedImage2Name
            # 4:  Image3Name (or DoNotUse)
            # 5:  AlignedImage3Name
            # 6:  AlignMethod
            # 7:  AlternateImage1 (aligned similarly to Image2)
            # 8:  AlternateAlignedImage1
            # 9:  AlternateImage2
            # 10: AlternateAlignedImage2
            # 11: Wants cropping.
            new_setting_values = list(setting_values[:4])
            if (setting_values[4] != cps.DO_NOT_USE and
                setting_values[5] != cps.DO_NOT_USE):
                new_setting_values += [ setting_values[4], setting_values[5],
                                        A_SEPARATELY]
            for i in (7,9):
                if (setting_values[i] != cps.DO_NOT_USE and
                    setting_values[i+1] != cps.DO_NOT_USE):
                    new_setting_values += [setting_values[i], 
                                           setting_values[i+1],
                                           A_SIMILARLY]
            new_setting_values += [setting_values[6], setting_values[11]]
            setting_values = new_setting_values
            from_matlab = False
            variable_revision_number = 1
        elif from_matlab and variable_revision_number == 6:
            #
            # The Matlab align module has the following layout
            # 0:  Image1Name
            # 1:  AlignedImage1Name
            # 2:  Image2Name
            # 3:  AlignedImage2Name
            # 4:  Image3Name (or DoNotUse)
            # 5:  AlignedImage3Name
            # 6:  AlignMethod
            # 7:  AlternateImage1 (aligned similarly to Image2)
            # 8:  AlternateAlignedImage1
            # 9:  AlternateImage2
            # 10: AlternateAlignedImage2
            # 11: MoreImageName3
            # 12: MoreAlignedImageName3
            # 13: MoreImageName4
            # 14: MoreAlignedImageName4
            # 15: Wants cropping.
            new_setting_values = list(setting_values[:4])
            if (setting_values[4] != cps.DO_NOT_USE and
                setting_values[5] != cps.DO_NOT_USE):
                new_setting_values += [ setting_values[4], setting_values[5],
                                        A_SEPARATELY]
            for i in (7,9,11,13):
                if (setting_values[i] != cps.DO_NOT_USE and
                    setting_values[i+1] != cps.DO_NOT_USE):
                    new_setting_values += [setting_values[i], 
                                           setting_values[i+1],
                                           A_SIMILARLY]
            new_setting_values += [setting_values[6], setting_values[11]]
            setting_values = new_setting_values
            from_matlab = False
            variable_revision_number = 1
            
        if (not from_matlab) and variable_revision_number == 1:
            # Moved final settings (alignment method, cropping) to the top
            setting_values = (setting_values[-2:] + setting_values[:-2])
            variable_revision_number = 2
            
        return setting_values, variable_revision_number, from_matlab

def offset_slice(pixels1, pixels2, i, j):
    '''Return two sliced arrays where the first slice is offset by i,j
    relative to the second slice.
    
    '''
    if i < 0:
        p1_imin = -i
        p1_imax = pixels1.shape[0]
        p2_imin = 0
        p2_imax = pixels1.shape[0]+i
    else:
        p1_imin = 0
        p1_imax = pixels1.shape[0]-i
        p2_imin = i
        p2_imax = pixels1.shape[0]
    if j < 0:
        p1_jmin = -j
        p1_jmax = pixels1.shape[1]
        p2_jmin = 0
        p2_jmax = pixels1.shape[1]+j
    else:
        p1_jmin = 0
        p1_jmax = pixels1.shape[1]-j
        p2_jmin = j
        p2_jmax = pixels1.shape[1]
    p1 = pixels1[p1_imin:p1_imax,p1_jmin:p1_jmax]
    p2 = pixels2[p2_imin:p2_imax,p2_jmin:p2_jmax]
    return (p1,p2)
    
def cumsum_quadrant(x, i_forwards, j_forwards):
    '''Return the cumulative sum going in the i, then j direction
    
    x - the matrix to be summed
    i_forwards - sum from 0 to end in the i direction if true
    j_forwards - sum from 0 to end in the j direction if true
    '''
    if i_forwards:
        x=x.cumsum(0)
    else:
        x=np.flipud(np.flipud(x).cumsum(0))
    if j_forwards:
        return x.cumsum(1)
    else:
        return np.fliplr(np.fliplr(x).cumsum(1))
    
def entropy(x):
    '''The entropy of x as if x is a probability distribution'''
    histogram = scind.histogram(x.astype(float), np.min(x), np.max(x), 256)
    n = np.sum(histogram)
    if n > 0 and np.max(histogram) > 0:
        histogram = histogram[histogram!=0]
        return np.log2(n) - np.sum(histogram * np.log2(histogram))/n
    else:
        return 0

def entropy2(x,y):
    '''Joint entropy of paired samples X and Y'''
    #
    # Bin each image into 256 gray levels
    #
    x = (stretch(x) * 255).astype(int)
    y = (stretch(y) * 255).astype(int)
    #
    # create an image where each pixel with the same X & Y gets
    # the same value
    #
    xy = 256*x+y
    xy = xy.flatten()
    sparse = scipy.sparse.coo_matrix((np.ones(xy.shape), 
                                      (xy,np.zeros(xy.shape))))
    histogram = sparse.toarray()
    n=np.sum(histogram)
    if n > 0 and np.max(histogram) > 0:
        histogram = histogram[histogram>0]
        return np.log2(n) - np.sum(histogram * np.log2(histogram)) / n
    else:
        return 0
