function handles = IdentifySecondary(handles)

% Help for the Identify Secondary module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Identifies objects (e.g. cell edges) using "seed" objects identified by
% an Identify Primary module (e.g. nuclei).
% *************************************************************************
%
% This module identifies secondary objects (e.g. cell edges) based on two
% inputs: (1) a previous module's identification of primary objects (e.g.
% nuclei) and (2) an image stained for the secondary objects (not required
% for the Distance - N option). Each primary object is assumed to be completely
% within a secondary object (e.g. nuclei are completely within cells
% stained for actin).
%
% It accomplishes two tasks:
% (a) finding the dividing lines between secondary objects which touch each
% other. Three methods are available: Propagation, Watershed (an older
% version of Propagation), and Distance.
% (b) finding the dividing lines between the secondary objects and the
% background of the image. This is done by thresholding the image stained
% for secondary objects, except when using Distance - N.
%
% Settings:
%
% Methods to identify secondary objects:
% * Propagation - For task (a), this method will find dividing lines
% between clumped objects where the image stained for secondary objects
% shows a change in staining (i.e. either a dimmer or a brighter line).
% Smoother lines work better, but unlike the watershed method, small gaps
% are tolerated. This method is considered an improvement on the
% traditional watershed method. The dividing lines between objects are
% determined by a combination of the distance to the nearest primary object
% and intensity gradients. This algorithm uses local image similarity to
% guide the location of boundaries between cells. Boundaries are
% preferentially placed where the image's local appearance changes
% perpendicularly to the boundary. Reference: TR Jones, AE Carpenter, P
% Golland (2005) Voronoi-Based Segmentation of Cells on Image Manifolds,
% ICCV Workshop on Computer Vision for Biomedical Image Applications, pp.
% 535-543. For task (b), thresholding is used.
%
% * Watershed - For task (a), this method will find dividing lines between
% objects by looking for dim lines between objects. For task (b),
% thresholding is used. Reference: Vincent, Luc, and Pierre Soille,
% "Watersheds in Digital Spaces: An Efficient Algorithm Based on Immersion
% Simulations," IEEE Transactions of Pattern Analysis and Machine
% Intelligence, Vol. 13, No. 6, June 1991, pp. 583-598.
%
% * Distance - This method is bit unusual because the edges of the primary
% objects are expanded a specified distance to create the secondary
% objects. For example, if nuclei are labeled but there is no stain to help
% locate cell edges, the nuclei can simply be expanded in order to estimate
% the cell's location. This is often called the 'doughnut' or 'annulus' or
% 'ring' approach for identifying the cytoplasmic compartment. Using the
% Distance - N method, the image of the secondary staining is not used at
% all, and these expanded objects are the final secondary objects. Using
% the Distance - B method, thresholding is used to eliminate background
% regions from the secondary objects. This allows the extent of the
% secondary objects to be limited to a certain distance away from the edge
% of the primary objects.
%
% Select automatic thresholding method or enter an absolute threshold:
%    The threshold affects the stringency of the lines between the objects
% and the background. You can have the threshold automatically calculated
% using several methods, or you can enter an absolute number between 0 and
% 1 for the threshold (to see the pixel intensities for your images in the
% appropriate range of 0 to 1, use the CellProfiler Image Tool,
% 'ShowOrHidePixelData', in a window showing your image). There are
% advantages either way. An absolute number treats every image identically,
% but is not robust to slight changes in lighting/staining conditions
% between images. An automatically calculated threshold adapts to changes
% in lighting/staining conditions between images and is usually more
% robust/accurate, but it can occasionally produce a poor threshold for
% unusual/artifactual images. It also takes a small amount of time to
% calculate.
%    The threshold which is used for each image is recorded as a
% measurement in the output file, so if you find unusual measurements from
% one of your images, you might check whether the automatically calculated
% threshold was unusually high or low compared to the other images.
%    There are four methods for finding thresholds automatically, Otsu's
% method, the Mixture of Gaussian (MoG) method, the Background method, and
% the Ridler-Calvard method. The Otsu method uses our version of the Matlab
% function graythresh (the code is in the CellProfiler subfunction
% CPthreshold). Our modifications include taking into account the max and
% min values in the image and log-transforming the image prior to
% calculating the threshold. Otsu's method is probably better if you don't
% know anything about the image, or if the percent of the image covered by
% objects varies substantially from image to image. But if you know the
% object coverage percentage and it does not vary much from image to image,
% the MoG can be better, especially if the coverage percentage is not near
% 50%. Note, however, that the MoG function is experimental and has not
% been thoroughly validated. The background function is very simple and is
% appropriate for images in which most of the image is background. It finds
% the mode of the histogram of the image, which is assumed to be the
% background of the image, and chooses a threshold at twice that value
% (which you can adjust with a Threshold Correction Factor, see below).
% This can be very helpful, for example, if your images vary in overall
% brightness but the objects of interest are always twice (or actually, any
% constant) as bright as the background of the image. The Ridler-Calvard
% method is simple and its results are often very similar to Otsu's. It
% chooses and initial threshold, and then iteratively calculates the next
% one by taking the mean of the average intensities of the background and
% foreground pixels determined by the first threshold, repeating this until
% the threshold converges.
%    You can also choose between global and adaptive thresholding, where
% global means that one threshold is used for the entire image and adaptive
% means that the threshold varies across the image. Adaptive is slower to
% calculate but provides more accurate edge determination.
%
% Threshold correction factor:
% When the threshold is calculated automatically, it may consistently be
% too stringent or too lenient. You may need to enter an adjustment factor
% which you empirically determine is suitable for your images. The number 1
% means no adjustment, 0 to 1 makes the threshold more lenient and greater
% than 1 (e.g. 1.3) makes the threshold more stringent. For example, the
% Otsu automatic thresholding inherently assumes that 50% of the image is
% covered by objects. If a larger percentage of the image is covered, the
% Otsu method will give a slightly biased threshold that may have to be
% corrected using a threshold correction factor.
%
% Lower and upper bounds on threshold:
% Can be used as a safety precaution when the threshold is calculated
% automatically. For example, if there are no objects in the field of view,
% the automatic threshold will be unreasonably low. In such cases, the
% lower bound you enter here will override the automatic threshold.
%
% Approximate percentage of image covered by objects:
% An estimate of how much of the image is covered with objects. This
% information is currently only used in the MoG (Mixture of Gaussian)
% thresholding but may be used for other thresholding methods in the future
% (see below).
%
% Regularization factor (for propagation method only):
% This method takes two factors into account when deciding where to draw
% the dividing line between two touching secondary objects: the distance to
% the nearest primary object, and the intensity of the secondary object
% image. The regularization factor controls the balance between these two
% considerations: A value of zero means that the distance to the nearest
% primary object is ignored and the decision is made entirely on the
% intensity gradient between the two competing primary objects. Larger
% values weight the distance between the two values more and more heavily.
% The regularization factor can be infinitely large, but around 10 or so,
% the intensity image is almost completely ignored and the dividing line
% will simply be halfway between the two competing primary objects.
%
% Note: Primary identify modules produce two (hidden) output images that
% are used by this module. The Segmented image contains the final, edited
% primary objects (i.e. objects at the border and those that are too small
% or large have been excluded). The SmallRemovedSegmented image is the
% same except that the objects at the border and the large objects have
% been included. These extra objects are used to perform the identification
% of secondary object outlines, since they are probably real objects (even
% if we don't want to measure them). Small objects are not used at this
% stage because they are more likely to be artifactual, and so they
% therefore should not "claim" any secondary object pixels.
%
% TECHNICAL DESCRIPTION OF THE PROPAGATION OPTION:
% Propagate labels from LABELS_IN to LABELS_OUT, steered by IMAGE and
% limited to MASK. MASK should be a logical array. LAMBDA is a
% regularization parameter, larger being closer to Euclidean distance in
% the image plane, and zero being entirely controlled by IMAGE. Propagation
% of labels is by shortest path to a nonzero label in LABELS_IN. Distance
% is the sum of absolute differences in the image in a 3x3 neighborhood,
% combined with LAMBDA via sqrt(differences^2 + LAMBDA^2). Note that there
% is no separation between adjacent areas with different labels (as there
% would be using, e.g., watershed). Such boundaries must be added in a
% postprocess. IdentifySecPropagateSubfunction is the subfunction
% implemented in C and MEX to perform the propagate algorithm.
%
% IdentifySecPropagateSubfunction.cpp is the source code, in C++
% IdentifySecPropagateSubfunction.dll is compiled for windows
% IdentifySecPropagateSubfunction.mexmac is compiled for macintosh
% IdentifySecPropagateSubfunction.mexglx is compiled for linux
% IdentifySecPropagateSubfunction.mexa64 is compiled for 64-bit linux
%
% To compile IdentifySecPropagateSubfunction for different operating
% systems, you will need to log on to that operating system and at the
% command line of MATLAB enter:
% mex IdentifySecPropagateSubfunction
%
% See also Identify primary modules.

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
%
% Website: http://www.cellprofiler.org
%
% $Revision: 1808 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%%% Sets up loop for test mode.
if strcmp(char(handles.Settings.VariableValues{CurrentModuleNum,12}),'Yes')
    IdentChoiceList = {'Distance - N' 'Distance - B' 'Watershed' 'Propagation'};
else
    IdentChoiceList = {char(handles.Settings.VariableValues{CurrentModuleNum,3})};
end

%textVAR01 = What did you call the primary objects you want to create secondary objects around?
%infotypeVAR01 = objectgroup
PrimaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the objects identified by this module?
%defaultVAR02 = Cells
%infotypeVAR02 = objectgroup indep
SecondaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Select the method to identify the secondary objects (Distance - B uses background; Distance - N does not):
%choiceVAR03 = Propagation
%choiceVAR03 = Watershed
%choiceVAR03 = Distance - N
%choiceVAR03 = Distance - B
%inputtypeVAR03 = popupmenu
OriginalIdentChoice = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What did you call the images to be used to find the edges of the secondary objects? For DISTANCE - N, this will not affect object identification, only the final display.
%infotypeVAR04 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Select an automatic thresholding method or enter an absolute threshold in the range [0,1]. To choose a binary image, select "Other" and type its name.  Choosing 'All' will use the Otsu Global method to calculate a single threshold for the entire image group. The other methods calculate a threshold for each image individually. Set interactively will allow you to manually adjust the threshold during the first cycle to determine what will work well.
%choiceVAR05 = Otsu Global
%choiceVAR05 = Otsu Adaptive
%choiceVAR05 = MoG Global
%choiceVAR05 = MoG Adaptive
%choiceVAR05 = Background Global
%choiceVAR05 = Background Adaptive
%choiceVAR05 = RidlerCalvard Global
%choiceVAR05 = RidlerCalvard Adaptive
%choiceVAR05 = All
%choiceVAR05 = Set interactively
Threshold = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu custom

%textVAR06 = Threshold correction factor
%defaultVAR06 = 1
ThresholdCorrection = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = Lower and upper bounds on threshold, in the range [0,1]
%defaultVAR07 = 0,1
ThresholdRange = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = For MoG thresholding, what is the approximate percentage of image covered by objects?
%choiceVAR08 = 10%
%choiceVAR08 = 20%
%choiceVAR08 = 30%
%choiceVAR08 = 40%
%choiceVAR08 = 50%
%choiceVAR08 = 60%
%choiceVAR08 = 70%
%choiceVAR08 = 80%
%choiceVAR08 = 90%
pObject = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = For DISTANCE, enter the number of pixels by which to expand the primary objects [Positive integer]
%defaultVAR09 = 10
DistanceToDilate = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,9}));

%textVAR10 = For PROPAGATION, enter the regularization factor (0 to infinity). Larger=distance,0=intensity
%defaultVAR10 = 0.05
RegularizationFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,10}));

%textVAR11 = What do you want to call the outlines of the identified objects (optional)?
%defaultVAR11 = Do not save
%infotypeVAR11 = outlinegroup indep
SaveOutlines = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%textVAR12 = Do you want to run in test mode where each method for identifying secondary objects is compared?
%choiceVAR12 = No
%choiceVAR12 = Yes
TestMode = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','CheckScale');

%%% Retrieves the preliminary label matrix image that contains the primary
%%% segmented objects which have only been edited to discard objects
%%% that are smaller than a certain size.  This image
%%% will be used as markers to segment the secondary objects with this
%%% module.  Checks first to see whether the appropriate image exists.
PrelimPrimaryLabelMatrixImage = CPretrieveimage(handles,['SmallRemovedSegmented', PrimaryObjectName],ModuleName,'DontCheckColor','DontCheckScale',size(OrigImage));

%%% Retrieves the label matrix image that contains the edited primary
%%% segmented objects which will be used to weed out which objects are
%%% real - not on the edges and not below or above the specified size
%%% limits. Checks first to see whether the appropriate image exists.
EditedPrimaryLabelMatrixImage = CPretrieveimage(handles,['Segmented', PrimaryObjectName],ModuleName,'DontCheckColor','DontCheckScale',size(OrigImage));

%%% Checks if a custom entry was selected for Threshold
if ~(strncmp(Threshold,'Otsu',4) || strncmp(Threshold,'MoG',3) || strncmp(Threshold,'Background',10) || strncmp(Threshold,'RidlerCalvard',13) || strcmp(Threshold,'All') || strcmp(Threshold,'Set interactively'))
    if isnan(str2double(Threshold))
        GetThreshold = 0;
        BinaryInputImage = CPretrieveimage(handles,Threshold,ModuleName,'MustBeGray','CheckScale');
    else
        GetThreshold = 1;
    end
else
    GetThreshold = 1;
end

%%% Checks that the Min and Max threshold bounds have valid values
index = strfind(ThresholdRange,',');
if isempty(index)
    error(['Image processing was canceled in the ', ModuleName, ' module because the Min and Max threshold bounds are invalid.'])
end

MinimumThreshold = ThresholdRange(1:index-1);
MaximumThreshold = ThresholdRange(index+1:end);

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% STEP 1: Marks at least some of the background by applying a
%%% weak threshold to the original image of the secondary objects.
if GetThreshold
    [handles,Threshold] = CPthreshold(handles,Threshold,pObject,MinimumThreshold,MaximumThreshold,ThresholdCorrection,OrigImage,ImageName,ModuleName,SecondaryObjectName);
else Threshold = 0; % should never be used
end
%%% ANNE REPLACED THIS LINE 11-06-05.
%%% Thresholds the original image.
% ThresholdedOrigImage = im2bw(OrigImage, Threshold);

%%% Thresholds the original image.
if GetThreshold
    ThresholdedOrigImage = OrigImage > Threshold;
else
    ThresholdedOrigImage = logical(BinaryInputImage);
end
Threshold = mean(Threshold(:));       % Use average threshold downstreams

for IdentChoiceNumber = 1:length(IdentChoiceList)

    IdentChoice = IdentChoiceList{IdentChoiceNumber};

    if strncmp(IdentChoice,'Distance',8)
        if strcmp(IdentChoice(12),'N')
            %%% Creates the structuring element using the user-specified size.
            StructuringElement = strel('disk', DistanceToDilate);
            %%% Dilates the preliminary label matrix image (edited for small only).
            DilatedPrelimSecObjectLabelMatrixImage = imdilate(PrelimPrimaryLabelMatrixImage, StructuringElement);
            %%% Converts to binary.
            DilatedPrelimSecObjectBinaryImage = im2bw(DilatedPrelimSecObjectLabelMatrixImage,.5);
            %%% Computes nearest neighbor image of nuclei centers so that the dividing
            %%% line between secondary objects is halfway between them rather than
            %%% favoring the primary object with the greater label number.
            [ignore, Labels] = bwdist(full(PrelimPrimaryLabelMatrixImage>0)); %#ok We want to ignore MLint error checking for this line.
            drawnow
            %%% Remaps labels in Labels to labels in PrelimPrimaryLabelMatrixImage.
            if max(Labels(:)) == 0,
                Labels = ones(size(Labels));
            end
            ExpandedRelabeledDilatedPrelimSecObjectImage = PrelimPrimaryLabelMatrixImage(Labels);
            %%% Removes the background pixels (those not labeled as foreground in the
            %%% DilatedPrelimSecObjectBinaryImage). This is necessary because the
            %%% nearest neighbor function assigns *every* pixel to a nucleus, not just
            %%% the pixels that are part of a secondary object.
            RelabeledDilatedPrelimSecObjectImage = zeros(size(ExpandedRelabeledDilatedPrelimSecObjectImage));
            RelabeledDilatedPrelimSecObjectImage(DilatedPrelimSecObjectBinaryImage) = ExpandedRelabeledDilatedPrelimSecObjectImage(DilatedPrelimSecObjectBinaryImage);
            drawnow
        elseif strcmp(IdentChoice(12),'B')
            [labels_out,d]=IdentifySecPropagateSubfunction(PrelimPrimaryLabelMatrixImage,OrigImage,ThresholdedOrigImage,1.0);
            labels_out(d>DistanceToDilate) = 0;
            labels_out((PrelimPrimaryLabelMatrixImage > 0)) = PrelimPrimaryLabelMatrixImage((PrelimPrimaryLabelMatrixImage > 0));
            RelabeledDilatedPrelimSecObjectImage = labels_out;
        end

        EditedPrimaryBinaryImage = im2bw(EditedPrimaryLabelMatrixImage,.5);

        %%% Removes objects that are not in the edited EditedPrimaryLabelMatrixImage.
        LookUpTable = sortrows(unique([PrelimPrimaryLabelMatrixImage(:) EditedPrimaryLabelMatrixImage(:)],'rows'),1);
        b=zeros(max(LookUpTable(:,1)+1),2);
        b(LookUpTable(:,1)+1,1)=LookUpTable(:,1);
        b(LookUpTable(:,1)+1,2)=LookUpTable(:,2);
        b(:,1) = 0:size(b,1)-1;
        LookUpColumn = b(:,2);
        FinalLabelMatrixImage = LookUpColumn(RelabeledDilatedPrelimSecObjectImage+1);

    elseif strcmp(IdentChoice,'Propagation')
        %%% STEP 2: Starting from the identified primary objects, the secondary
        %%% objects are identified using the propagate function, written by Thouis
        %%% R. Jones. Calls the function
        %%% "IdentifySecPropagateSubfunction.mexmac" (or whichever version is
        %%% appropriate for the computer platform being used), which consists of C
        %%% code that has been compiled to run quickly within Matlab.
        PropagatedImage = IdentifySecPropagateSubfunction(PrelimPrimaryLabelMatrixImage,OrigImage,ThresholdedOrigImage,RegularizationFactor);
        drawnow

        %%% STEP 3: Remove objects that are not desired, edited objects.  The
        %%% edited primary object image is used rather than the preliminary one, so
        %%% that objects whose nuclei are on the edge of the image and who are
        %%% larger or smaller than the specified size are discarded.
        %%% Converts the EditedPrimaryBinaryImage to binary.
        EditedPrimaryBinaryImage = im2bw(EditedPrimaryLabelMatrixImage,.5);
        %%% Finds the locations and labels for different regions.
        area_locations2 = find(PropagatedImage);
        area_labels2 = PropagatedImage(area_locations2);
        drawnow
        %%% Creates a sparse matrix with column as label and row as location,
        %%% with the value of the center at (I,J) if location I has label J.
        %%% Taking the maximum of this matrix gives the largest valued center
        %%% overlapping a particular label.  Tacking on a zero and pushing
        %%% labels through the resulting map removes any background regions.
        map2 = [0 full(max(sparse(area_locations2, area_labels2, EditedPrimaryLabelMatrixImage(area_locations2))))];
        HoleyPrelimLabelMatrixImage = map2(PropagatedImage + 1);
        %%% Fills in holes in the HoleyPrelimLabelMatrixImage image.
        %%% Filters the image for maxima (Plus sign neighborhood, ignoring zeros).
        MaximaImage = ordfilt2(HoleyPrelimLabelMatrixImage, 5, [0 1 0; 1 1 1 ; 0 1 0]);
        %%% This is a pain.  On sparse matrices, min returns zero almost always
        %%% (because the matrices are mostly zero, of course).  So we need to invert
        %%% the labels so we can use max to find the minimum adjacent label as well,
        %%% below.  This also takes care of boundaries, which otherwise return zero
        %%% in the min filter.
        LargestLabelImage = max(HoleyPrelimLabelMatrixImage(:));
        TempImage = HoleyPrelimLabelMatrixImage;
        TempImage(HoleyPrelimLabelMatrixImage > 0) = LargestLabelImage - TempImage(HoleyPrelimLabelMatrixImage > 0) + 1;
        %%% Filters the image for minima (Plus sign neighborhood).
        MinimaImage = ordfilt2(TempImage, 5, [0 1 0; 1 1 1 ; 0 1 0]);
        %%% Marks and labels the zero regions.
        ZeroRegionImage = CPclearborder(bwlabel(HoleyPrelimLabelMatrixImage==0, 4));
        drawnow
        %%% Uses sparse matrices to find the minimum and maximum label adjacent
        %%% to each zero-region.
        ZeroLocations = find(ZeroRegionImage);
        ZeroLabels = ZeroRegionImage(ZeroLocations);
        MinByRegion = full(max(sparse(ZeroLocations, ZeroLabels, MinimaImage(ZeroLocations))));
        %%% Remaps to correct order (see above).
        MinByRegion = LargestLabelImage - MinByRegion + 1;
        MaxByRegion = full(max(sparse(ZeroLocations, ZeroLabels, MaximaImage(ZeroLocations))));
        %%% Anywhere the min and max are the same is a region surrounded by a
        %%% single value.
        Surrounded = (MinByRegion == MaxByRegion);
        %%% Creates a map that turns a labelled zero-region into the surrounding
        %%% label if it's surrounded, and into zero if it's not surrounded.
        %%% (Pad by a leading zero so 0 maps to 0 when 1 is added.)
        Remap = [ 0 (Surrounded .* MinByRegion)];
        ZeroRegionImage = Remap(ZeroRegionImage + 1);
        %%% Now all surrounded zeroregions should have been remapped to their
        %%% new value, or zero if not surrounded.
        PrelimLabelMatrixImage = max(HoleyPrelimLabelMatrixImage, ZeroRegionImage);
        drawnow

        %%% STEP 4: Relabels the final objects so that their numbers
        %%% correspond to the numbers used for nuclei.
        %%% For each object, one label and one label location is acquired and
        %%% stored.
        [LabelsUsed,LabelLocations] = unique(EditedPrimaryLabelMatrixImage);
        %%% The +1 increment accounts for the fact that there are zeros in the
        %%% image, while the LabelsUsed starts at 1.
        LabelsUsed(PrelimLabelMatrixImage(LabelLocations(2:end))+1) = EditedPrimaryLabelMatrixImage(LabelLocations(2:end));
        FinalLabelMatrixImage = LabelsUsed(PrelimLabelMatrixImage+1);
    elseif strcmp(IdentChoice,'Watershed')
        %%% In order to use the watershed transform to find dividing lines between
        %%% the secondary objects, it is necessary to identify the foreground
        %%% objects and to identify a portion of the background.  The foreground
        %%% objects are retrieved as the binary image of primary objects from the
        %%% previously run image analysis module.   This forces the secondary
        %%% object's outline to extend at least as far as the edge of the primary
        %%% objects.

        %%% Inverts the image.
        InvertedThresholdedOrigImage = imcomplement(ThresholdedOrigImage);

        %%% NOTE: There are two other ways to mark the background prior to
        %%% watershedding; I think the method used above is best, but I have
        %%% included the ideas for two alternate methods.
        %%% METHOD (2): Threshold the original image (or a smoothed image)
        %%% so that background pixels are black.  This is overly strong, so instead
        %%% of weakly thresholding the image as is done in METHOD (1),  you can then "thin"
        %%% the background pixels by computing the SKIZ
        %%% (skeleton of influence zones), which is done by watershedding the
        %%% distance transform of the thresholded image.  These watershed lines are
        %%% then superimposed on the marked image that will be watershedded to
        %%% segment the objects.  I think this would not produce results different
        %%% from METHOD 1 (the one used above), since METHOD 1 overlays the
        %%% outlines of the primary objects anyway.
        %%% This method is based on the Mathworks Image Processing Toolbox demo
        %%% "Marker-Controlled Watershed Segmentation".  I found it online; I don't
        %%% think it is in the Matlab Demos that are found through help.  It uses
        %%% an image of a box of oranges.
        %%%
        %%% METHOD (3):  (I think this method does not work well for clustered
        %%% objects.)  The distance transformed image containing the marked objects
        %%% is watershedded, which produces lines midway between the marked
        %%% objects.  These lines are superimposed on the marked image that will be
        %%% watershedded to segment the objects. But if marked objects are
        %%% clustered and not a uniform distance from each other, this will produce
        %%% background lines on top of actual objects.
        %%% This method is based on Gonzalez, et al. Digital Image Processing using
        %%% Matlab, page 422-425.

        %%% STEP 2: Identify the outlines of each primary object, so that each
        %%% primary object can be definitely separated from the background.  This
        %%% solves the problem of some primary objects running
        %%% right up against the background pixels and therefore getting skipped.
        %%% Note: it is less accurate and less fast to use edge detection (sobel)
        %%% to identify the edges of the primary objects.
        drawnow
        %%% Converts the PrelimPrimaryLabelMatrixImage to binary.
        PrelimPrimaryBinaryImage = im2bw(PrelimPrimaryLabelMatrixImage,.5);
        %%% Creates the structuring element that will be used for dilation.
        StructuringElement = strel('square',3);
        %%% Dilates the Primary Binary Image by one pixel (8 neighborhood).
        DilatedPrimaryBinaryImage = imdilate(PrelimPrimaryBinaryImage, StructuringElement);
        %%% Subtracts the PrelimPrimaryBinaryImage from the DilatedPrimaryBinaryImage,
        %%% which leaves the PrimaryObjectOutlines.
        PrimaryObjectOutlines = DilatedPrimaryBinaryImage - PrelimPrimaryBinaryImage;

        %%% STEP 3: Produce the marker image which will be used for the first
        %%% watershed.
        drawnow
        %%% Combines the foreground markers and the background markers.
        BinaryMarkerImagePre = PrelimPrimaryBinaryImage | InvertedThresholdedOrigImage;
        %%% Overlays the PrimaryObjectOutlines to maintain distinctions between each
        %%% primary object and the background.
        BinaryMarkerImage = BinaryMarkerImagePre;
        BinaryMarkerImage(PrimaryObjectOutlines == 1) = 0;

        %%% STEP 4: Calculate the Sobel image, which reflects gradients, which will
        %%% be used for the watershedding function.
        drawnow
        %%% Calculates the 2 sobel filters.  The sobel filter is directional, so it
        %%% is used in both the horizontal & vertical directions and then the
        %%% results are combined.
        filter1 = fspecial('sobel');
        filter2 = filter1';
        %%% Applies each of the sobel filters to the original image.
        I1 = imfilter(OrigImage, filter1);
        I2 = imfilter(OrigImage, filter2);
        %%% Adds the two images.
        %%% The Sobel operator results in negative values, so the absolute values
        %%% are calculated to prevent errors in future steps.
        AbsSobeledImage = abs(I1) + abs(I2);

        %%% STEP 5: Perform the first watershed.
        drawnow

        %%% Overlays the foreground and background markers onto the
        %%% absolute value of the Sobel Image, so there are black nuclei on top of
        %%% each dark object, with black background.
        Overlaid = imimposemin(AbsSobeledImage, BinaryMarkerImage);
        %%% Perform the watershed on the marked absolute-value Sobel Image.
        BlackWatershedLinesPre = watershed(Overlaid);
        %%% Bug workaround (see step 9).
        BlackWatershedLinesPre2 = im2bw(BlackWatershedLinesPre,.5);
        BlackWatershedLines = bwlabel(BlackWatershedLinesPre2);

        %%% STEP 6: Identify and extract the secondary objects, using the watershed
        %%% lines.
        drawnow
        %%% The BlackWatershedLines image is a label matrix where the watershed
        %%% lines = 0 and each distinct object is assigned a number starting at 1.
        %%% This image is converted to a binary image where all the objects = 1.
        SecondaryObjects1 = im2bw(BlackWatershedLines,.5);
        %%% Identifies objects in the binary image using bwlabel.
        %%% Note: Matlab suggests that in some circumstances bwlabeln is faster
        %%% than bwlabel, even for 2D images.  I found that in this case it is
        %%% about 10 times slower.
        LabelMatrixImage1 = bwlabel(SecondaryObjects1,4);
        drawnow

        %%% STEP 7: Discarding background "objects".  The first watershed function
        %%% simply divides up the image into regions.  Most of these regions
        %%% correspond to actual objects, but there are big blocks of background
        %%% that are recognized as objects. These can be distinguished from actual
        %%% objects because they do not overlap a primary object.

        %%% The following changes all the labels in LabelMatrixImage1 to match the
        %%% centers they enclose (from PrelimPrimaryBinaryImage), and marks as background
        %%% any labeled regions that don't overlap a center. This function assumes
        %%% that every center is entirely contained in one labeled area.  The
        %%% results if otherwise may not be well-defined. The non-background labels
        %%% will be renumbered according to the center they enclose.

        %%% Finds the locations and labels for different regions.
        area_locations = find(LabelMatrixImage1);
        area_labels = LabelMatrixImage1(area_locations);
        %%% Creates a sparse matrix with column as label and row as location,
        %%% with the value of the center at (I,J) if location I has label J.
        %%% Taking the maximum of this matrix gives the largest valued center
        %%% overlapping a particular label.  Tacking on a zero and pushing
        %%% labels through the resulting map removes any background regions.
        map = [0 full(max(sparse(area_locations, area_labels, PrelimPrimaryBinaryImage(area_locations))))];
        ActualObjectsBinaryImage = map(LabelMatrixImage1 + 1);

        %%% STEP 8: Produce the marker image which will be used for the second
        %%% watershed.
        drawnow
        %%% The module has now produced a binary image of actual secondary
        %%% objects.  The gradient (Sobel) image was used for watershedding, which
        %%% produces very nice divisions between objects that are clumped, but it
        %%% is too stringent at the edges of objects that are isolated, and at the
        %%% edges of clumps of objects. Therefore, the stringently identified
        %%% secondary objects are used as markers for a second round of
        %%% watershedding, this time based on the original (intensity) image rather
        %%% than the gradient image.

        %%% Creates the structuring element that will be used for dilation.
        StructuringElement = strel('square',3);
        %%% Dilates the Primary Binary Image by one pixel (8 neighborhood).
        DilatedActualObjectsBinaryImage = imdilate(ActualObjectsBinaryImage, StructuringElement);
        %%% Subtracts the PrelimPrimaryBinaryImage from the DilatedPrimaryBinaryImage,
        %%% which leaves the PrimaryObjectOutlines.
        ActualObjectOutlines = DilatedActualObjectsBinaryImage - ActualObjectsBinaryImage;
        %%% Produces the marker image which will be used for the watershed. The
        %%% foreground markers are taken from the ActualObjectsBinaryImage; the
        %%% background markers are taken from the same image as used in the first
        %%% round of watershedding: InvertedThresholdedOrigImage.
        BinaryMarkerImagePre2 = ActualObjectsBinaryImage | InvertedThresholdedOrigImage;
        %%% Overlays the ActualObjectOutlines to maintain distinctions between each
        %%% secondary object and the background.
        BinaryMarkerImage2 = BinaryMarkerImagePre2;
        BinaryMarkerImage2(ActualObjectOutlines == 1) = 0;

        %%% STEP 9: Perform the second watershed.
        %%% As described above, the second watershed is performed on the original
        %%% intensity image rather than on a gradient (Sobel) image.
        drawnow
        %%% Inverts the original image.
        InvertedOrigImage = imcomplement(OrigImage);
        %%% Overlays the foreground and background markers onto the
        %%% InvertedOrigImage, so there are black secondary object markers on top
        %%% of each dark secondary object, with black background.
        MarkedInvertedOrigImage = imimposemin(InvertedOrigImage, BinaryMarkerImage2);
        %%% Performs the watershed on the MarkedInvertedOrigImage.
        SecondWatershedPre = watershed(MarkedInvertedOrigImage);
        %%% BUG WORKAROUND:
        %%% There is a bug in the watershed function of Matlab that often results in
        %%% the label matrix result having two objects labeled with the same label.
        %%% I am not sure whether it is a bug in how the watershed image is
        %%% produced (it seems so: the resulting objects often are nowhere near the
        %%% regional minima) or whether it is simply a problem in the final label
        %%% matrix calculation. Matlab has been informed of this issue and has
        %%% confirmed that it is a bug (February 2004). I think that it is a
        %%% reasonable fix to convert the result of the watershed to binary and
        %%% remake the label matrix so that each label is used only once. In later
        %%% steps, inappropriate regions are weeded out anyway.
        SecondWatershedPre2 = im2bw(SecondWatershedPre,.5);
        SecondWatershed = bwlabel(SecondWatershedPre2);
        drawnow

        %%% STEP 10: As in step 7, remove objects that are actually background
        %%% objects.  See step 7 for description. This time, the edited primary object image is
        %%% used rather than the preliminary one, so that objects whose nuclei are
        %%% on the edge of the image and who are larger or smaller than the
        %%% specified size are discarded.

        %%% Converts the EditedPrimaryBinaryImage to binary.
        EditedPrimaryBinaryImage = im2bw(EditedPrimaryLabelMatrixImage,.5);
        %%% Finds the locations and labels for different regions.
        area_locations2 = find(SecondWatershed);
        area_labels2 = SecondWatershed(area_locations2);
        %%% Creates a sparse matrix with column as label and row as location,
        %%% with the value of the center at (I,J) if location I has label J.
        %%% Taking the maximum of this matrix gives the largest valued center
        %%% overlapping a particular label.  Tacking on a zero and pushing
        %%% labels through the resulting map removes any background regions.
        map2 = [0 full(max(sparse(area_locations2, area_labels2, EditedPrimaryBinaryImage(area_locations2))))];
        FinalBinaryImagePre = map2(SecondWatershed + 1);
        %%% Fills holes in the FinalBinaryPre image.
        FinalBinaryImage = imfill(FinalBinaryImagePre, 'holes');
        %%% Converts the image to label matrix format. Even if the above step
        %%% is excluded (filling holes), it is still necessary to do this in order
        %%% to "compact" the label matrix: this way, each number corresponds to an
        %%% object, with no numbers skipped.
        ActualObjectsLabelMatrixImage3 = bwlabel(FinalBinaryImage);
        %%% The final objects are relabeled so that their numbers
        %%% correspond to the numbers used for nuclei.
        %%% For each object, one label and one label location is acquired and
        %%% stored.
        [LabelsUsed,LabelLocations] = unique(EditedPrimaryLabelMatrixImage);
        %%% The +1 increment accounts for the fact that there are zeros in the
        %%% image, while the LabelsUsed starts at 1.
        LabelsUsed(ActualObjectsLabelMatrixImage3(LabelLocations(2:end))+1) = EditedPrimaryLabelMatrixImage(LabelLocations(2:end));
        FinalLabelMatrixImagePre = LabelsUsed(ActualObjectsLabelMatrixImage3+1);
        %%% The following is a workaround for what seems to be a bug in the
        %%% watershed function: very very rarely two nuclei end up sharing one
        %%% "cell" object, so that one of the nuclei ends up without a
        %%% corresponding cell.  I am trying to determine why this happens exactly.
        %%% When the cell is measured, the area (and other
        %%% measurements) are recorded as [], which causes problems when dependent
        %%% measurements (e.g. perimeter/area) are attempted.  It results in divide
        %%% by zero errors and the mean area = NaN and so on.  So, the Primary
        %%% label matrix image (where it is nonzero) is written onto the Final cell
        %%% label matrix image pre so that every primary object has at least some
        %%% pixels of secondary object.
        FinalLabelMatrixImage = FinalLabelMatrixImagePre;
        FinalLabelMatrixImage(EditedPrimaryLabelMatrixImage ~= 0) = EditedPrimaryLabelMatrixImage(EditedPrimaryLabelMatrixImage ~= 0);
    end

    %%% Calculates the ColoredLabelMatrixImage for displaying in the figure
    %%% window in subplot(2,2,2).
    ColoredLabelMatrixImage = CPlabel2rgb(handles,FinalLabelMatrixImage);
    %%% Calculates OutlinesOnOrigImage for displaying in the figure
    %%% window in subplot(2,2,3).
    %%% Note: these outlines are not perfectly accurate; for some reason it
    %%% produces more objects than in the original image.  But it is OK for
    %%% display purposes.
    %%% Maximum filters the image with a 3x3 neighborhood.
    MaxFilteredImage = ordfilt2(FinalLabelMatrixImage,9,ones(3,3),'symmetric');
    %%% Determines the outlines.
    IntensityOutlines = FinalLabelMatrixImage - MaxFilteredImage;
    %%% Converts to logical.
    warning off MATLAB:conversionToLogical
    LogicalOutlines = logical(IntensityOutlines);
    warning on MATLAB:conversionToLogical
    %%% Determines the grayscale intensity to use for the cell outlines.
    LineIntensity = max(OrigImage(:));
    %%% Overlays the outlines on the original image.
    ObjectOutlinesOnOrigImage = OrigImage;
    ObjectOutlinesOnOrigImage(LogicalOutlines) = LineIntensity;
    %%% Calculates BothOutlinesOnOrigImage for displaying in the figure
    %%% window in subplot(2,2,4).
    %%% Creates the structuring element that will be used for dilation.
    StructuringElement = strel('square',3);
    %%% Dilates the Primary Binary Image by one pixel (8 neighborhood).
    DilatedPrimaryBinaryImage = imdilate(EditedPrimaryBinaryImage, StructuringElement);
    %%% Subtracts the PrelimPrimaryBinaryImage from the DilatedPrimaryBinaryImage,
    %%% which leaves the PrimaryObjectOutlines.
    PrimaryObjectOutlines = DilatedPrimaryBinaryImage - EditedPrimaryBinaryImage;
    BothOutlinesOnOrigImage = ObjectOutlinesOnOrigImage;
    BothOutlinesOnOrigImage(PrimaryObjectOutlines == 1) = LineIntensity;

    if strcmp(TestMode,'Yes')
        SecondaryTestFig = findobj('Tag','SecondaryTestFigure');
        if isempty(SecondaryTestFig)
            SecondaryTestFig = CPfigure(handles,'Image','Tag','SecondaryTestFigure','Name','Secondary Test Figure');
        else
            CPfigure(handles,'Image',SecondaryTestFig);
        end
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            CPresizefigure(ObjectOutlinesOnOrigImage,'TwoByTwo',SecondaryTestFig);
        end
        subplot(2,2,IdentChoiceNumber);
        CPimagesc(ObjectOutlinesOnOrigImage,handles);
        title(IdentChoiceList(IdentChoiceNumber));
    end

    if strcmp(OriginalIdentChoice,IdentChoice)
        if ~isfield(handles.Measurements,SecondaryObjectName)
            handles.Measurements.(SecondaryObjectName) = {};
        end

        if ~isfield(handles.Measurements,PrimaryObjectName)
            handles.Measurements.(PrimaryObjectName) = {};
        end

        handles = CPrelateobjects(handles,SecondaryObjectName,PrimaryObjectName,FinalLabelMatrixImage,EditedPrimaryLabelMatrixImage);

        %%%%%%%%%%%%%%%%%%%%%%%
        %%% DISPLAY RESULTS %%%
        %%%%%%%%%%%%%%%%%%%%%%%
        drawnow

        ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
        if any(findobj == ThisModuleFigureNumber)
            %%% Activates the appropriate figure window.
            CPfigure(handles,'Image',ThisModuleFigureNumber);
            if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
                CPresizefigure(OrigImage,'TwoByTwo',ThisModuleFigureNumber);
            end
            ObjectCoverage = 100*sum(sum(FinalLabelMatrixImage > 0))/numel(FinalLabelMatrixImage);
            uicontrol(ThisModuleFigureNumber,'Style','Text','Units','Normalized','Position',[0.25 0.01 .6 0.04],...
                'BackgroundColor',[.7 .7 .9],'HorizontalAlignment','Left','String',sprintf('Threshold:  %0.3f               %0.1f%% of image consists of objects',Threshold,ObjectCoverage),'FontSize',handles.Preferences.FontSize);
            %%% A subplot of the figure window is set to display the original image.
            subplot(2,2,1); 
            CPimagesc(OrigImage,handles); 
            title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
            %%% A subplot of the figure window is set to display the colored label
            %%% matrix image.
            subplot(2,2,2); 
            CPimagesc(ColoredLabelMatrixImage,handles); 
            title(['Outlined ',SecondaryObjectName]);
            %%% A subplot of the figure window is set to display the original image
            %%% with secondary object outlines drawn on top.
            subplot(2,2,3); 
            CPimagesc(ObjectOutlinesOnOrigImage,handles); 
            title([SecondaryObjectName, ' Outlines on Input Image']);
            %%% A subplot of the figure window is set to display the original
            %%% image with outlines drawn for both the primary and secondary
            %%% objects.
            subplot(2,2,4); 
            CPimagesc(BothOutlinesOnOrigImage,handles); 
            title(['Outlines of ', PrimaryObjectName, ' and ', SecondaryObjectName, ' on Input Image']);
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% SAVE DATA TO HANDLES STRUCTURE %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        drawnow

        %%% Saves the final, segmented label matrix image of secondary objects to
        %%% the handles structure so it can be used by subsequent modules.
        fieldname = ['Segmented',SecondaryObjectName];
        handles.Pipeline.(fieldname) = FinalLabelMatrixImage;

        %%% TODO: why do we have the same thing twice here, with an OR?
        if strcmp(IdentChoice,'Propagation') || strcmp(IdentChoice,'Propagation')
            %%% Saves the Threshold value to the handles structure.
            %%% Storing the threshold is a little more complicated than storing other measurements
            %%% because several different modules will write to the handles.Measurements.Image.Threshold
            %%% structure, and we should therefore probably append the current threshold to an existing structure.
            % First, if the Threshold fields don't exist, initialize them
            if ~isfield(handles.Measurements.Image,'ThresholdFeatures')
                handles.Measurements.Image.ThresholdFeatures = {};
                handles.Measurements.Image.Threshold = {};
            end
            %%% Search the ThresholdFeatures to find the column for this object type
            column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ThresholdFeatures,SecondaryObjectName)));
            %%% If column is empty it means that this particular object has not been segmented before. This will
            %%% typically happen for the first cycle. Append the feature name in the
            %%% handles.Measurements.Image.ThresholdFeatures matrix
            if isempty(column)
                handles.Measurements.Image.ThresholdFeatures(end+1) = {SecondaryObjectName};
                column = length(handles.Measurements.Image.ThresholdFeatures);
            end
            handles.Measurements.Image.Threshold{handles.Current.SetBeingAnalyzed}(1,column) = Threshold;
        end

        %%% Saves the ObjectCount, i.e. the number of segmented objects.
        if ~isfield(handles.Measurements.Image,'ObjectCountFeatures')
            handles.Measurements.Image.ObjectCountFeatures = {};
            handles.Measurements.Image.ObjectCount = {};
        end
        column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ObjectCountFeatures,SecondaryObjectName)));
        if isempty(column)
            handles.Measurements.Image.ObjectCountFeatures(end+1) = {SecondaryObjectName};
            column = length(handles.Measurements.Image.ObjectCountFeatures);
        end
        handles.Measurements.Image.ObjectCount{handles.Current.SetBeingAnalyzed}(1,column) = max(FinalLabelMatrixImage(:));

        %%% Saves the location of each segmented object
        handles.Measurements.(SecondaryObjectName).LocationFeatures = {'CenterX','CenterY'};
        tmp = regionprops(FinalLabelMatrixImage,'Centroid');
        Centroid = cat(1,tmp.Centroid);
        handles.Measurements.(SecondaryObjectName).Location(handles.Current.SetBeingAnalyzed) = {Centroid};

        %%% Saves images to the handles structure so they can be saved to the hard
        %%% drive, if the user requested.
        try
            if ~strcmpi(SaveOutlines,'Do not save')
                handles.Pipeline.(SaveOutlines) = LogicalOutlines;
            end
        catch error(['The object outlines were not calculated by the ', ModuleName, ' module, so these images were not saved to the handles structure. The Save Images module will therefore not function on these images. This is just for your information - image processing is still in progress, but the Save Images module will fail if you attempted to save these images.'])
        end
    end
end