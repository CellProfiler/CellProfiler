function handles = Average(handles)

% Help for the Make Projection/Average Images module:
% Category: Image Processing
%
% This module makes a projection of a set of images (e.g. a Z-stack)
% by averaging the pixel intensities at each pixel position.
%
% How it works:
% This module works by averaging together all of the images. The first
% time through the pipeline (i.e. for image set 1), the whole set of
% images (as defined by a Load Images module) is used to calculate one
% projected image. Subsequent runs through the pipeline (i.e. for
% image set 2 through the end) produce no new results, but processing
% is not aborted in case other modules are being run for some reason.
% The projection image calculated the first time through the pipeline
% is still available to other modules during subsequent runs through
% the pipeline.
%
% Settings:
%
% Enter L or P:
% If you choose L, the module will calculate the single, averaged
% projection image the first time through the pipeline by loading
% every image of the type specified in the Load Images module. It is
% then acceptable to use the resulting image later in the pipeline. If
% you choose P, the module will allow the pipeline to cycle through
% all of the image sets.  With this option, the module does not need
% to follow a Load Images module; it is acceptable to make the single,
% averaged projection from images resulting from other image
% processing steps in the pipeline. However, the resulting projection
% image will not be available until the last image set has been
% processed, so it cannot be used in subsequent modules unless they
% are instructed to wait until the last image set.
%
% SAVING IMAGES: The image produced by this module can be easily saved
% using the Save Images module, using the name you assign.
%
% See also CORRECTILLUMINATION_APPLY,
% CORRECTILLUMINATION_CALCULATEUSINGINTENSITIES.

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
drawnow



%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the images to be averaged to make the projection?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the resulting projection image?
%infotypeVAR02 = imagegroup indep
%defaultVAR02 = ProjectedBlue
ProjectionImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Are the images you want to use to be loaded straight from a Load Images module (L), or are they being produced by the pipeline (P)? See the help for details.
%choiceVAR03 = Load Images module
%choiceVAR03 = Pipeline
SourceIsLoadedOrPipeline = char(handles.Settings.VariableValues{CurrentModuleNum,3});
SourceIsLoadedOrPipeline = SourceIsLoadedOrPipeline(1);
%inputtypeVAR03 = popupmenu

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow



ReadyFlag = 'Not Ready';
try
    if strncmpi(SourceIsLoadedOrPipeline, 'L',1) == 1 && handles.Current.SetBeingAnalyzed == 1
        %%% The first time the module is run, the projection image is
        %%% calculated.
        [ProjectionImage, ReadyFlag] = CPaverageimages(handles, 'DoNow', ImageName, 'ignore');
    elseif strncmpi(SourceIsLoadedOrPipeline, 'P',1) == 1
        [ProjectionImage, ReadyFlag] = CPaverageimages(handles, 'Accumulate', ImageName, ProjectionImageName);
    else
        error('Image processing was canceled because you must choose either "L" or "P" in the Make Projection/Average Images module');
    end
catch [ErrorMessage, ErrorMessage2] = lasterr;
    error(['An error occurred in the Correct Illumination_Calculate Using Intensities module. Matlab says the problem is: ', ErrorMessage, ErrorMessage2])
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow


fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;

    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(2) = originalsize(2) + originalsize(4)/2;
        newsize(3) = originalsize(3)/2;
        newsize(4) = originalsize(4)/2;
        set(ThisModuleFigureNumber, 'position', newsize);
        drawnow
    end
    if strncmpi(SourceIsLoadedOrPipeline, 'L',1) == 1 && handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        %%% The projection image is displayed the first time through
        %%% the set. For subsequent image sets, this figure is not
        %%% updated at all, to prevent the need to load the projection
        %%% image from the handles structure.
        %%% Activates the appropriate figure window.
        CPfigure(handles,ThisModuleFigureNumber);
        imagesc(ProjectionImage);
        title(['Final Projection Image, based on all ', num2str(NumberOfImages), ' images']);
        colormap(gray)
    elseif strncmpi(SourceIsLoadedOrPipeline, 'P',1) == 1
        %%% The accumulated projection image so far is displayed each time through
        %%% the pipeline.
        %%% Activates the appropriate figure window.
        CPfigure(handles,ThisModuleFigureNumber);
        imagesc(ProjectionImage);
        title(['Projection Image so far, based on Image set # 1 - ', num2str(handles.Current.SetBeingAnalyzed)]);
        colormap(gray)
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow





%%% If running in non-cycling mode (straight from the hard drive using
%%% a LoadImages module), the projection image and its flag need only
%%% be saved to the handles structure after the first image set is
%%% processed. If running in cycling mode (Pipeline mode), the
%%% projection image and its flag are saved to the handles structure
%%% after every image set is processed.
if strncmpi(SourceIsLoadedOrPipeline, 'P',1) == 1 | (strncmpi(SourceIsLoadedOrPipeline, 'L',1) == 1 && handles.Current.SetBeingAnalyzed == 1)
    %%% Saves the projected image to the handles structure so it can be used by
    %%% subsequent modules.
    handles.Pipeline.(ProjectionImageName) = ProjectionImage;
    %%% Saves the ready flag to the handles structure so it can be used by
    %%% subsequent modules.
    fieldname = [ProjectionImageName,'ReadyFlag'];
    handles.Pipeline.(fieldname) = ReadyFlag;
end
