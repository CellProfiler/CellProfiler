function handles = Flip(handles)

% Help for the Flip module:
% Category: Image Processing
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
% $Revision: 1725 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

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
    error(['Image processing has been canceled. Prior to running the Flip module, you must have previously run a module to load an image. You specified in the Flip module that this image was called ', ImageName, ' which should have produced a field in the handles structure called ', fieldname, '. The IdentifyPrimAutomatic module cannot find this image.']);
end
OrigImage = handles.Pipeline.(fieldname);

if strcmp(LeftToRight,'No') && strcmp(TopToBottom,'No')
    error('You are not flipping the image!');
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%

FlippedImage = OrigImage;
[Rows,Columns,Dimensions] = size(FlippedImage);

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
if any(findobj == ThisModuleFigureNumber) == 1;
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        %%% Sets the window to be half as wide as usual.
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 250;
        set(ThisModuleFigureNumber, 'position', newsize);
    end

    drawnow
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); imagesc(OrigImage);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    subplot(2,1,2); imagesc(FlippedImage); title('Flipped Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(OutputName) = FlippedImage;