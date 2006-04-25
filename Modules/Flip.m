function handles = Flip(handles)

% Help for the Flip module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Flips an image from top to bottom, left to right, or both.
% *************************************************************************

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
%   Susan Ma
%
% Website: http://www.cellprofiler.org
%
% $Revision: 1725 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image you want to flip?
%infotypeVAR01 = imagegroup
%inputtypeVAR01 = popupmenu
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the flipped image?
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
OrigImage = CPretrieveimage(handles,ImageName,ModuleName);

if strcmp(LeftToRight,'No') && strcmp(TopToBottom,'No')
    error(['Image processing was canceled in the ', ModuleName, ' module because with the current settings you have not chosen to flip the image.']);
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

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
    end
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1);
    CPimagesc(OrigImage,handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    subplot(2,1,2);
    CPimagesc(FlippedImage,handles);
    title('Flipped Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(OutputName) = FlippedImage;