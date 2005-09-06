function handles = ProcessOutlines(handles)

% Help for the Process Outlines module:
% Category: Image Processing
%
% Sorry, help does not yet exist for this module.  We wrote it really
% quickly for a collaborator.
%
% SAVING IMAGES: The images of the objects produced by this module can
% be easily saved using the Save Images module using the name:
% Segmented + whatever you called the objects (e.g. SegmentedCells).
% This will be a grayscale image where each object is a different
% intensity. If you want to save other intermediate images, alter the
% code for this module to save those images to the handles structure
% (see the SaveImages module help) and then use the Save Images
% module.
%
% See also <nothing relevant>

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$




drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%



%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the images you want to process?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the objects identified by this module?
%defaultVAR02 = ProcessedOutlines
%infotypeVAR02 = objectgroup indep
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Enter the threshold (Positive number, Max = 1):
%defaultVAR03 = 0.05
Threshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));
%textVAR04 = Note: this module may fill in holes between objects that are not desired, so follow it with an identify primary objects module.

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
%%% Checks whether the image exists in the handles structure.
    if isfield(handles.Pipeline, ImageName) == 0
    error(['Image processing has been canceled. Prior to running the Identify Primary Intensity module, you must have previously run a module to load an image. You specified in the Identify Primary Intensity module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', ImageName, '. The Identify Primary Intensity module cannot find this image.']);
    end
OrigImage = handles.Pipeline.(ImageName);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the Identify Primary Intensity module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow



BinaryImage = im2bw(imcomplement(OrigImage),Threshold);
FilledImage = imfill(BinaryImage,'holes');
ObjectsIdentifiedImage = imsubtract(FilledImage,BinaryImage);

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow



fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;

    drawnow
    CPfigure(handles,ThisModuleFigureNumber);
    %%% Sets the width of the figure window to be appropriate (half width).
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 0.5*originalsize(3);
        set(ThisModuleFigureNumber, 'position', newsize);
    end
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); imagesc(OrigImage);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,1,2); imagesc(ObjectsIdentifiedImage); title(['Processed ',ObjectName]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Saves the processed image to the handles structure.
fieldname = ['Segmented',ObjectName];
handles.Pipeline.(fieldname) = ObjectsIdentifiedImage;
