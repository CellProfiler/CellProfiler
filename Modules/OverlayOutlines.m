function handles = OverlayOutlines(handles)

% Help for the Overlay Outlines module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Places outlines produced by an identify module over a desired image.
% *************************************************************************
%
% Sorry, this module has not yet been documented.
%
% See also n/a.

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
% $Revision: 1718 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = On what image would you like to display the outlines?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the outlines that you would like to display?
%infotypeVAR02 = outlinegroup
OutlineName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = Would you like to set the outlines to be the maximum in the image, or the maximum possible value for this image format?
%choiceVAR03 = Max of image
%choiceVAR03 = Max possible
MaxType = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = What do you want to call the image with the outlines displayed?
%defaultVAR04 = Do not save
%infotypeVAR04 = imagegroup indep
SavedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

try
    OrigImage = handles.Pipeline.(ImageName);
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because the image ' ImageName ' could not be found. Perhaps there is a typo?']);
end

try
    OutlineImage = handles.Pipeline.(OutlineName);
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because the image ' OutlineName ' could not be found. Perhaps there is a typo?  Make sure you saved the outlines in an earlier module.']);
end

if any(size(OrigImage,1) ~= size(OutlineImage,1)) ||  any(size(OrigImage,2) ~= size(OutlineImage,2))
    error(['Image processing was canceled in the ', ModuleName, ' module because the size of the image, ' num2str(size(OrigImage)) ' is not the same as the size of the outlines, ' num2str(size(OutlineImage))]);
end

if size(OrigImage,3) ~= 3
    if strcmp(MaxType,'Max of image')
        ValueToUseForOutlines = max(max(OrigImage));
    elseif strcmp(MaxType,'Max possible')
        if isfloat(OrigImage(1,1))
            ValueToUseForOutlines=1;
        else
            ValueToUseForOutlines = intmax(class(OrigImage(1,1)));
        end
    else
        error(['Image processing was canceled in the ', ModuleName, ' module because the value of MaxType was not recognized.']);
    end

    NewImage = OrigImage;
    NewImage(OutlineImage ~= 0) = ValueToUseForOutlines;
else
    NewImage1 = OrigImage(:,:,1);
    NewImage2 = OrigImage(:,:,2);
    NewImage3 = OrigImage(:,:,3);
    NewImage1(OutlineImage ~= 0) = 1;
    NewImage2(OutlineImage ~= 0) = 1;
    NewImage3(OutlineImage ~= 0) = 1;
    NewImage(:,:,1) = NewImage1;
    NewImage(:,:,2) = NewImage2;
    NewImage(:,:,3) = NewImage3;
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;

    FigHandle = CPfigure(handles,ThisModuleFigureNumber);

    CPimagesc(NewImage);
    title(['Original Image with Outline Overlay, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    uicontrol(FigHandle,'units','normalized','position',[.01 .5 .06 .04],'string','off',...
        'UserData',{OrigImage NewImage},'backgroundcolor',[.7 .7 .9],...
        'Callback','string=get(gcbo,''string'');UserData=get(gcbo,''UserData''); if strcmp(string,''off''),imagesc(UserData{1});set(gcbo,''string'',''on'');elseif strcmp(string,''on''),imagesc(UserData{2});set(gcbo,''string'',''off'');else,set(gcbo,''string'',''on'');end;clear UserData string;');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if ~strcmp(SavedImageName,'Do not save')
    handles.Pipeline.(SavedImageName) = NewImage;
end