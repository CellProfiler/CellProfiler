function handles = MakeProjection(handles)

% Help for the Average module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Makes a projection either by averaging or taking the maximum pixel value
% at each pixel position.
%
% *************************************************************************
%
% This module averages a set of images by averaging the pixel intensities
% at each pixel position. When this module is used to average a Z-stack
% (3-D image stack), this process is known as making a projection.
%
% Settings:
%
% * What did you call the images to be made into a projection?:
%   Choose an image from among those loaded by a module or created by the
% pipeline, which will be made into a projection with the corresponding images of every
% image set.
%
% * What kind of projection would you like to make?:
%   If you choose Average, the average pixel intensity at each pixel
%   position will be used to created the final image.  If you choose
%   Maximum, the maximum pixel value at each pixel position will be used to
%   created the final image.
% * What do you want to call the projected image?:
%   Give a name to the resulting image, which could be used in subsequent
% modules. See the next setting for restrictions.
%
% * Are the images you want to use to be loaded straight from a Load Images
% module, or are they being produced by the pipeline?:
%   If you choose Load Images Module, the module will calculate the single,
% projected image the first time through the pipeline (i.e. for cycle 1) by
% loading the image of the type specified above of every image set.
% It is then acceptable to use the resulting image
% later in the pipeline. Subsequent runs through the pipeline (i.e. for
% cycle 2 through the end) produce no new results. The projcted image
% calculated during the first cycle is still available to other modules
% during subsequent cycles.
%   If you choose Pipeline, the module will calculate the single, projected
% image during the last cycle of the pipeline. This is because it must wait
% for preceding modules in the pipeline to produce their results before it
% can calculate an projected image. For example, you cannot calculate the
% projection of all Cropped images until after the last image cycle completes
% and the last cropped image is produced. Note that in this mode, the
% resulting projected image will not be available until the last cycle has
% been processed, so the projected image it produces cannot be used in
% subsequent modules unless they are instructed to wait until the last
% cycle.
%
% See also CorrectIllumination_Calculate.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

% PyCP notes: The 4th variable is basically asking whether you want to do
% this calculation during the first cycle (in which case the resulting
% image will be available downstream in the pipeline) or whether you need
% to let other processing proceed cycle-by-cycle in order to gather the raw
% material for this module to do its thing, thus producing the final image
% only at the end of the last cycle. So this should be more clear because
% right now it seems like a weird question to ask the user where their
% images are coming from - of course in the new PyCP we could just figure
% this out automatically probably, but that's not exactly the point of the
% question.

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the images to be made into a projection?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What kind of projection would you like to make?
%choiceVAR02 = Average
%choiceVAR02 = Maximum
ProjectionType = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the projected image?
%defaultVAR03 = ProjectedBlue
%infotypeVAR03 = imagegroup indep
ProjectionImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Are the images you want to use to be loaded straight from a Load Images module, or are they being produced by the pipeline? 
%choiceVAR04 = Load Images module
%choiceVAR04 = Pipeline
SourceIsLoadedOrPipeline = char(handles.Settings.VariableValues{CurrentModuleNum,4});
SourceIsLoadedOrPipeline = SourceIsLoadedOrPipeline(1);
%inputtypeVAR04 = popupmenu


%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% If running in non-cycling mode (straight from the hard drive using a
%%% Load Images module), the averaged image and its flag need only be
%%% calculated and saved to the handles structure after the first cycle is
%%% processed. If running in cycling mode (Pipeline mode), the averaged
%%% image and its flag are saved to the handles structure after every cycle
%%% is processed.
if strncmpi(SourceIsLoadedOrPipeline, 'L',1) && handles.Current.SetBeingAnalyzed ~= 1
    return
end

ReadyFlag = 'Not Ready';
if strcmpi(ProjectionType,'Average')
    try
        if strncmpi(SourceIsLoadedOrPipeline, 'L',1)
            %%% If we are in Load Images mode, the averaged image is calculated
            %%% the first time the module is run.
            if  isfield(handles.Pipeline,['Pathname', ImageName]);
                [handles, ProjectionImage, ReadyFlag,ignore] = CPaverageimages(handles, 'DoNow', ImageName, 'ignore','ignore');
            else
                error(['Image processing was canceled in the ', ModuleName, ' module because CellProfiler could not look up the name of the folder where the ' ImageName ' images were loaded from.  This is most likely because this module is not using images that were loaded directly from the load images module. See help for more details.']);
            end
        elseif strncmpi(SourceIsLoadedOrPipeline, 'P',1)
            [handles, ProjectionImage, ReadyFlag, ignore] = CPaverageimages(handles, 'Accumulate', ImageName, ProjectionImageName,'ignore');
        else
            error(['Image processing was canceled in the ', ModuleName, ' module because you must choose either "Load images" or "Pipeline".']);
        end
    catch [ErrorMessage, ErrorMessage2] = lasterr;
        error(['An error occurred in the ', ModuleName, ' module. Matlab says the problem is: ', ErrorMessage, ErrorMessage2])
    end
elseif strcmpi(ProjectionType, 'Maximum')
    try
        if strncmpi(SourceIsLoadedOrPipeline,'L',1)
        fieldname = ['Pathname', ImageName];
        try Pathname = handles.Pipeline.(fieldname);
        catch error(['Image processing was canceled because CellProfiler could not find the input image.CellProfiler expected to find an image named "', ImageName, '" but that image has not been created by the pipeline. Please adjust your pipeline to produce the image "', ImageName, ''])
        end
        fieldname = ['FileList', ImageName];
        try FileList = handles.Pipeline.(fieldname);
        catch
            error(['Image processing was canceled because CellProfiler could not find the input image. CellProfiler expected to find an image named "', ImageName, '" but that image has not been created by the pipeline. Please adjust your pipeline to produce the image "', ImageName, ''])
        end
        OrigImage = cell(length(FileList),1);
        for i=1:length(FileList)
            OrigImage{i} = CPimread(fullfile(Pathname,char(FileList(i))));
        end
        ProjectionImage = OrigImage{1};
        %handles.Pipeline.(ProjectionImageName) = zeros(SizeOrig);
        for i = 2:length(FileList)
            ProjectionImage = max(ProjectionImage,OrigImage{i});
        end
        elseif strncmp(SourceIsLoadedOrPipeline,'P',1)
            fieldname = ['', ImageName];
            if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
                %%% Checks whether the image to be analyzed exists in the
                %%% handles structure.
                if isfield(handles.Pipeline, ImageName)==0,
                    error(['Image processing was cancelled because CellProfiler could not find the input image.  CellProfiler expected to find an image named "',ImageName,'" but that image has not been created by the pipeline.  Please adjust your pipeline to produce the image "',ImageName,''])
                end
                %%%First Image
                OrigImage = CPretrieveimage(handles,fieldname,ModuleName);
                %%% Creates the empty variable so it can be retrieved later
                %%% without causing an error on the first image set.
                handles = CPaddimages(handles,ProjectionImageName,zeros(size(OrigImage)));
            end
            %%%Current Image
            OrigImage = CPretrieveimage(handles,fieldname,ModuleName);
            %%%Projected Image so far
            ProjectionImage = CPretrieveimage(handles,ProjectionImageName,ModuleName);
            ProjectionImage = max(ProjectionImage,OrigImage);
        else
            error(['Image processing was canceled in the ', ModuleName, ' module because you must choose either "Load images" or "Pipeline".']);
        end
    catch [ErrorMessage, ErrorMessage2] = lasterr;
        error(['An error occurred in the ', ModuleName, ' module. Matlab says the problem is: ', ErrorMessage, ErrorMessage2])
    end
end


%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    ThisFigure = CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(ProjectionImage,'OneByOne',ThisModuleFigureNumber)
    end
    
    [ignore,hAx] = CPimagesc(ProjectionImage,handles,ThisFigure);
    if strncmpi(SourceIsLoadedOrPipeline, 'L',1)
        %%% The averaged image is displayed the first time through the set.
        %%% For subsequent cycles, this figure is not updated at all, to
        %%% prevent the need to load the averaged image from the handles
        %%% structure.
        title(hAx,['Final Projected Image, based on all ', num2str(handles.Current.NumberOfImageSets), ' images']);
    elseif strncmpi(SourceIsLoadedOrPipeline, 'P',1)
        %%% The accumulated averaged image so far is displayed each time
        %%% through the pipeline.
        title(hAx,['Projected Image so far, based on image # 1 - ', num2str(handles.Current.SetBeingAnalyzed)]);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the averaged image to the handles structure so it can be used by
%%% subsequent modules.
handles = CPaddimages(handles,ProjectionImageName,ProjectionImage);
%%% Saves the ready flag to the handles structure so it can be used by
%%% subsequent modules.
fieldname = [ProjectionImageName,'ReadyFlag'];
handles.Pipeline.(fieldname) = ReadyFlag;