function handles = Flip(handles)

% Help for the Flip module:
% Category: Image Processing
%
% SHORT DESCRIPTION: Flips an image from top to bottom, left to right, or
% both.
% *************************************************************************
%
% See also <nothing relevant>.

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
% $Revision: 1725 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

%textVAR01 = What is the input image?
%infotypeVAR01 = imagegroup
%inputtypeVAR01 = popupmenu
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the output image?
%defaultVAR02 = FlippedOrigBlue
%infotypeVAR02 = imagegroup indep
OutputName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Do you want to flip from left to right?
%choiceVAR03 = Yes
%choiceVAR03 = No
%inputtypeVAR03 = popupmenu
LeftToRight = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Do you want to flip from top to bottom?
%choiceVAR04 = Yes
%choiceVAR04 = No
%inputtypeVAR04 = popupmenu
TopToBottom = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
fieldname = ['',  ImageName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    error(['Image processing was canceled in the ', ModuleName, ' module. Prior to running the Flip module, you must have previously run a module to load an image. You specified in the Flip module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', fieldname, '. The IdentifyPrimAutomatic module cannot find this image.']);
end
OrigImage = handles.Pipeline.(fieldname);

if strcmp(LeftToRight,'No') && strcmp(TopToBottom,'No')
    error('You are not flipping the image!');
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

FlippedImage = OrigImage;
[Rows,Columns,Dimensions] = size(FlippedImage); %#ok

if strcmp(LeftToRight,'Yes')
    if Dimensions == 1
        FlippedImage = fliplr(FlippedImage);
    else
        FlippedImage(:,:,1) = fliplr(FlippedImage(:,:,1));
        FlippedImage(:,:,2) = fliplr(FlippedImage(:,:,2));
        FlippedImage(:,:,3) = fliplr(FlippedImage(:,:,3));
    end
end

if strcmp(TopToBottom,'Yes')
    if Dimensions == 1
        FlippedImage = flipud(FlippedImage);
    else
        FlippedImage(:,:,1) = flipud(FlippedImage(:,:,1));
        FlippedImage(:,:,2) = flipud(FlippedImage(:,:,2));
        FlippedImage(:,:,3) = flipud(FlippedImage(:,:,3));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber);

    %%% Sets the window to be half as wide as usual.
    originalsize = get(ThisModuleFigureNumber, 'position');
    newsize = originalsize;
    if newsize(3) ~= 250
        newsize(3) = 250;
        set(ThisModuleFigureNumber, 'position', newsize);
    end

    drawnow
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1);
    ImageHandle = imagesc(OrigImage);
    set(ImageHandle,'ButtonDownFcn','CPImageTool(gco)');
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    subplot(2,1,2);
    ImageHandle = imagesc(FlippedImage);
    set(ImageHandle,'ButtonDownFcn','CPImageTool(gco)');
    title('Flipped Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(OutputName) = FlippedImage;