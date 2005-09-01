function handles = PlaceAdjacent(handles)

% Help for the Place Adjacent module:
% Category: Image Processing
%
% This module places two images next to each other, either
% horizontally or vertically.
%
% SAVING IMAGES: The images produced by this module can be
% easily saved using the Save Images module, using the name you
% assign.
%
% See also <nothing relevant>.

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

%textVAR01 = What did you call the first image to be placed?
%infotypeVAR01 = imagegroup
ImageName1 = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the second image to be placed?
%infotypeVAR02 = imagegroup
ImageName2 = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the resulting image?
%defaultVAR03 = AdjacentImage
%infotypeVAR03 = imagegroup indep
AdjacentImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Placement Type.
%choiceVAR04 = Horizontal
%choiceVAR04 = Vertical
HorizontalOrVertical = char(handles.Settings.VariableValues{CurrentModuleNum,4});
HorizontalOrVertical = HorizontalOrVertical(1);
%inputtypeVAR04 = popupmenu

%textVAR05 = Can the incoming images be deleted from the pipeline after they are placed adjacent (this saves memory, but prevents you from using the incoming images later in the pipeline)?
%choiceVAR05 = No
%choiceVAR05 = Yes
DeletePipeline = char(handles.Settings.VariableValues{CurrentModuleNum,5});
DeletePipeline = DeletePipeline(1);
%inputtypeVAR05 = popupmenu

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
fieldname = ['', ImageName1];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Place Adjacent module could not find the input image.  It was supposed to be named ', ImageName1, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage1 = handles.Pipeline.(fieldname);

%%% Removes the image from the pipeline to save memory if requested.
if strncmpi(DeletePipeline,'Y',1) == 1
   handles.Pipeline = rmfield(handles.Pipeline,fieldname);
end
%%% Repeat for the second image.
fieldname = ['', ImageName2];
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing was canceled because the Place Adjacent module could not find the input image.  It was supposed to be named ', ImageName1, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
OrigImage2 = handles.Pipeline.(fieldname);
%%% Removes the image from the pipeline to save memory if requested.
if strncmpi(DeletePipeline,'Y',1) == 1
handles.Pipeline = rmfield(handles.Pipeline,fieldname);
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%



%%% Check that the images are the same height or width and place them
%%% adjacent to each other.

if strcmpi(HorizontalOrVertical,'H') == 1
    if size(OrigImage1,1) ~= size(OrigImage2,1)
        error('Image processing was canceled because the two input images must have the same height if they are to be placed horizontally adjacent to each other.')
    end
    %%% If one of the images is multidimensional (color), the other
    %%% one is replicated to match its dimensions.
    if size(OrigImage1,3) ~= size(OrigImage2,3)
        DesiredLayers = max(size(OrigImage1,3),size(OrigImage2,3));
        if size(OrigImage1,3) > size(OrigImage2,3)
            for i = 1:DesiredLayers, OrigImage2(:,:,i) = OrigImage2(:,:,1); end
        else
            for i = 1:DesiredLayers, OrigImage1(:,:,i) = OrigImage1(:,:,1); end
        end
    end
    AdjacentImage = cat(2,OrigImage1,OrigImage2);
elseif strcmpi(HorizontalOrVertical,'V') == 1
    if size(OrigImage1,2) ~= size(OrigImage2,2)
        error('Image processing was canceled because the two input images must have the same height if they are to be placed horizontally adjacent to each other.')
    end
    %%% If one of the images is multidimensional (color), the other
    %%% one is replicated to match its dimensions.
    if size(OrigImage1,3) ~= size(OrigImage2,3)
        DesiredLayers = max(size(OrigImage1,3),size(OrigImage2,3));
        if size(OrigImage1,3) > size(OrigImage2,3)
            for i = 1:DesiredLayers, OrigImage2(:,:,i) = OrigImage2(:,:,1); end
        else
            for i = 1:DesiredLayers, OrigImage1(:,:,i) = OrigImage1(:,:,1); end
        end
    end
    AdjacentImage = cat(1,OrigImage1,OrigImage2);
else
    error('Image processing was canceled because you must enter H or V to specify whether to place the images adjacent to each other horizontally or vertically.')
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow



fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;

    drawnow
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(OrigImage1);
    title(['First input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    subplot(2,2,3); imagesc(OrigImage2);
    title(['Second input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the Adjacent
    %%% Image.
    subplot(2,2,2); imagesc(AdjacentImage); title('Adjacent Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow





%%% Saves the Adjacent Image to the
%%% handles structure so it can be used by subsequent modules.
handles.Pipeline.(AdjacentImageName) = AdjacentImage;
