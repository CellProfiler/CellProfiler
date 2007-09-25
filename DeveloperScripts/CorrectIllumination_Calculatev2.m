function handles = CorrectIllumination_Calculatev2(handles)

% Help for the Correct Illumination Calculate module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Calculates an illumination function, used to correct uneven
% illumination/lighting/shading or to reduce uneven background in images.
% *************************************************************************
%
% This module calculates an illumination function which can be saved to the
% hard drive for later use (you should save in .mat format using the Save
% Images module), or it can be immediately applied to images later in the
% pipeline (using the CorrectIllumination_Apply module). This will correct
% for uneven illumination of each image.
%
% Illumination correction is challenging and we are writing a paper on it
% which should help clarify it (TR Jones, AE Carpenter, P Golland, in
% preparation). In the meantime, please be patient in trying to understand
% this module.
%
% Settings:
%
% * Regular or Background intensities?
%
% Regular intensities:
% If you have objects that are evenly dispersed across your image(s) and
% cover most of the image, then you can choose Regular intensities. Regular
% intensities makes the illumination function based on the intensity at
% each pixel of the image (or group of images if you are in All mode) and
% is most often rescaled (see below) and applied by division using
% CorrectIllumination_Apply. Note that if you are in Each mode or using a
% small set of images with few objects, there will be regions in the
% average image that contain no objects and smoothing by median filtering
% is unlikely to work well.
% Note: it does not make sense to choose (Regular + no smoothing + Each)
% because the illumination function would be identical to the original
% image and applying it will yield a blank image. You either need to smooth
% each image or you need to use All images.
%
% Background intensities:
% If you think that the background (dim points) between objects show the
% same pattern of illumination as your objects of interest, you can choose
% Background intensities. Background intensities finds the minimum pixel
% intensities in blocks across the image (or group of images if you are in
% All mode) and is most often applied by subtraction using the
% CorrectIllumination_Apply module.
% Note: if you will be using the Subtract option in the
% CorrectIllumination_Apply module, you almost certainly do NOT want to
% Rescale! See below!!
%
% * Each or All?
% Enter Each to calculate an illumination function for each image
% individually, or enter All to calculate the illumination function from
% all images at each pixel location. All is more robust, but depends on the
% assumption that the illumination patterns are consistent across all the
% images in the set and that the objects of interest are randomly
% positioned within each image. Applying illumination correction on each
% image individually may make intensity measures not directly comparable
% across different images.
%
% * Pipeline or Load Images?
% If you choose Load Images, the module will calculate the illumination
% correction function the first time through the pipeline by loading every
% image of the type specified in the Load Images module. It is then
% acceptable to use the resulting image later in the pipeline. If you
% choose Pipeline, the module will allow the pipeline to cycle through all
% of the cycles. With this option, the module does not need to follow a
% Load Images module; it is acceptable to make the single, averaged image
% from images resulting from other image processing steps in the pipeline.
% However, the resulting average image will not be available until the last
% cycle has been processed, so it cannot be used in subsequent modules
% unless they are instructed to wait until the last cycle.
%
% * Dilation:
% For some applications, the incoming images are binary and each object
% should be dilated with a gaussian filter in the final averaged
% (projection) image. This is for a sophisticated method of illumination
% correction where model objects are produced.
%
% * Smoothing Method:
% If requested, the resulting image is smoothed. See the help for the
% Smooth module for more details. If you are using Each mode, this is
% almost certainly necessary. If you have few objects in each image or a
% small image set, you may want to smooth. The goal is to smooth to the
% point where the illumination function resembles a believable pattern.
% That is, if it is a lamp illumination problem you are trying to correct,
% you would apply smoothing until you obtain a fairly smooth pattern
% without sharp bright or dim regions.  Note that smoothing is a
% time-consuming process, and fitting a polynomial is fastest but does not
% allow a very tight fit as compared to the slower median filtering method.
% Another option is to *completely* smooth the entire image by choosing
% "Smooth to average", which will create a flat, smooth image where every
% pixel of the image is the average of what the illumination function would
% otherwise have been.
%
% * Approximate width of objects:
% For certain smoothing methods, this will be used to calculate an adequate
% filter size. If you don't know the width of your objects, you can use the
% ShowOrHidePixelData image tool to find out or leave the word 'Automatic'
% to calculate a smoothing filter simply based on the size of the image.
%
%
% Rescaling:
% The illumination function can be rescaled so that the pixel intensities
% are all equal to or greater than one. This is recommended if you plan to
% use the division option in CorrectIllumination_Apply so that the
% corrected images are in the range 0 to 1. It is NOT recommended if you
% plan to use the Subtract option in CorrectIllumination_Apply! Note that
% as a result of the illumination function being rescaled from 1 to
% infinity, if there is substantial variation across the field of view, the
% rescaling of each image might be dramatic, causing the corrected images
% to be very dark.
%
% See also Average, CorrectIllumination_Apply, and Smooth modules.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%   Kyungnam Kim
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What type of images are being corrected?
%choiceVAR01 = Fluorescent
%choiceVAR01 = Brightfield
%choiceVAR01 = Phase Contrast
%choiceVAR01 = Other (see below)
Modality = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = If "Other"                               ...

%textVAR03 =      ... should the images be inverted before correcting?
%choiceVAR03 = No (do not invert)
%choiceVAR03 = Yes (invert)
Invert = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 =      ... should the images be log-transformed before correcting?
%choiceVAR04 = No (do not log-transform)
%choiceVAR04 = Yes (log-transform)
LogTransform = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = How smooth should the correction image be (filter radius in pixels)?
%defaultVAR05 = 50
SmoothingDiameter = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Enter Each to calculate an illumination correction for Each image individually or All to calculate an illumination correction based on All images to be corrected. See the help for details.
%choiceVAR06 = Each
%choiceVAR06 = All
EachOrAll = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = What is the first image (aka channel) to be corrected?
%infotypeVAR07 = imagegroup
ImageName1 = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = What is the second image (aka channel) to be corrected (or "None")?
%infotypeVAR08 = imagegroup
%choiceVAR08 = None
ImageName2 = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = What is the third image (aka channel) to be corrected (or "None")?
%infotypeVAR09 = imagegroup
%choiceVAR09 = None
ImageName3 = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR10 = What is the third image (aka channel) to be corrected (or "None")?
%infotypeVAR10 = imagegroup
%choiceVAR10 = None
ImageName4 = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%textVAR11 = What do you want to call the illumination correction image? (note: can only be saved in .mat format)
%defaultVAR11 = IllumCorrection
%infotypeVAR11 = imagegroup indep
IlluminationImageName = char(handles.Settings.VariableValues{CurrentModuleNum,11});
