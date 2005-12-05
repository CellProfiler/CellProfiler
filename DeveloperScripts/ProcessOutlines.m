function handles = ProcessOutlines(handles)

% Help for the Process Outlines module:
% Category: Image Processing
% 
% Takes an image with hand-drawn outlines and produces objects based
% on the outlines. It is useful for validation and when hand-outlining
% is necessary to accurately identify objects. The incoming outlined
% image can be hand drawn (e.g. using a marker on a transparency and
% scanning in the transparency) or it can be drawn in a program like
% photoshop.
%
% SETTINGS:
% Note that sophisticated options are not offered for thresholding,
% because the outlined image fed to this module should be essentially
% a black and white image (Dark background with white outlines). ??? IS
% THIS CORRECT?
%
% See also <nothing relevant>

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision: 2782 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow


[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the images you want to process?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the objects identified by this module?
%defaultVAR02 = Nuclei
%infotypeVAR02 = objectgroup indep
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Enter the threshold (Positive number, Max = 1):
%defaultVAR03 = 0.05
Threshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));
%textVAR04 = Note: this module may fill in holes between objects that are not desired, so follow it with an identify primary objects module.

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable.
%%% Checks whether the image exists in the handles structure.
    if isfield(handles.Pipeline, ImageName) == 0
    error(['Image processing was canceled in the ', ModuleName, ' module. Prior to running this module, you must have previously run a module to load an image. You specified that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', ImageName, '. The module cannot find this image.']);
    end
OrigImage = handles.Pipeline.(ImageName);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error(['Image processing was canceled in the ', ModuleName, ' module because it requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.'])
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

BinaryImage = im2bw(imcomplement(OrigImage),Threshold);
FilledImage = imfill(BinaryImage,'holes');
ObjectsIdentifiedImage = imsubtract(FilledImage,BinaryImage);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber) == 1;
    drawnow
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    %%% Sets the width of the figure window to be appropriate (half width).
    %%% TODO: update to new resizefigure subfunction!!
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 0.5*originalsize(3);
        set(ThisModuleFigureNumber, 'position', newsize);
    end
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); CPimagesc(OrigImage,handles.Preferences.IntensityColorMap);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,1,2); CPimagesc(ObjectsIdentifiedImage,handles.Preferences.IntensityColorMap); 
    title(['Processed ',ObjectName]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the processed image to the handles structure.
fieldname = ['Segmented',ObjectName];
handles.Pipeline.(fieldname) = ObjectsIdentifiedImage;