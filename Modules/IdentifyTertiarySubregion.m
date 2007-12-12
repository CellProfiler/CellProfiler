function handles = IdentifyTertiarySubregion(handles)

% Help for the Identify Tertiary Subregion module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Identifies tertiary obects (e.g. cytoplasm) by removing the primary
% objects (e.g. nuclei) from secondary objects (e.g. cells) leaving a
% ring shape.
% *************************************************************************
%
% This module will take the smaller identified objects and remove from them
% the larger identified objects. For example, "subtracting" the nuclei from
% the cells will leave just the cytoplasm, the properties of which can then
% be measured by Measure modules. The larger objects should therefore be
% equal in size or larger than the smaller objects and must completely
% contain the smaller objects.  Both inputs should be objects produced by
% identify modules, not images.
%
% Note: creating subregions using this module can result in objects that
% are not contiguous, which does not cause problems when running the
% Measure Intensity and Texture modules, but does cause problems when
% running the Measure Area Shape module because calculations of the
% perimeter, aspect ratio, solidity, etc. cannot be made for noncontiguous
% objects.
%
% See also Identify Primary and Identify Secondary modules.

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
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the larger identified objects?
%infotypeVAR01 = objectgroup
SecondaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the smaller identified objects?
%infotypeVAR02 = objectgroup
PrimaryObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the new subregions?
%defaultVAR03 = Cytoplasm
%infotypeVAR03 = objectgroup indep
SubregionObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What do you want to call the outlines of the identified objects (optional)?
%defaultVAR04 = Do not save
%infotypeVAR04 = outlinegroup indep
SaveOutlines = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
PrimaryObjectImage = CPretrieveimage(handles,['Segmented', PrimaryObjectName],ModuleName,'MustBeGray','DontCheckScale');

%%% Retrieves the Secondary object segmented image.
SecondaryObjectImage = CPretrieveimage(handles,['Segmented', SecondaryObjectName],ModuleName,'MustBeGray','DontCheckScale');

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Erodes the primary object image and then subtracts it from the
%%% secondary object image.  This prevents the subregion from having zero
%%% pixels (which cannot be measured in subsequent measure modules) in the
%%% cases where the secondary object is exactly the same size as the
%%% primary object.
%%%
%%% WARNING: THIS MEANS TERTIARY REGIONS ARE NOT EXCLUSIVE... PRIMARY +
%%% TERTIARY ~= SECONDARY

%%% For the cases where one of the label matrices was produced from a
%%% cropped image, the sizes of the matrices will not be equal, so the
%%% line above will fail. So, we crop the LabelMatrix and try again to
%%% see if the matrices are then the proper size. Removes Rows and
%%% Columns that are completely blank.
if any(size(SecondaryObjectImage) < size(PrimaryObjectImage))
    ColumnTotals = sum(PrimaryObjectImage,1);
    RowTotals = sum(PrimaryObjectImage,2)';
    warning off all
    ColumnsToDelete = ~logical(ColumnTotals);
    RowsToDelete = ~logical(RowTotals);
    warning on all
    drawnow
    CroppedLabelMatrix = PrimaryObjectImage;
    CroppedLabelMatrix(:,ColumnsToDelete,:) = [];
    CroppedLabelMatrix(RowsToDelete,:,:) = [];
    clear PrimaryObjectImage
    PrimaryObjectImage = CroppedLabelMatrix;
    %%% In case the entire image has been cropped away, we store a single
    %%% zero pixel for the variable.
    if isempty(PrimaryObjectImage)
        PrimaryObjectImage = 0;
    end
elseif any(size(SecondaryObjectImage) > size(PrimaryObjectImage))
    ColumnTotals = sum(SecondaryObjectImage,1);
    RowTotals = sum(SecondaryObjectImage,2)';
    warning off all
    ColumnsToDelete = ~logical(ColumnTotals);
    RowsToDelete = ~logical(RowTotals);
    warning on all
    drawnow
    CroppedLabelMatrix = SecondaryObjectImage;
    CroppedLabelMatrix(:,ColumnsToDelete,:) = [];
    CroppedLabelMatrix(RowsToDelete,:,:) = [];
    clear SecondaryObjectImage
    SecondaryObjectImage = CroppedLabelMatrix;
    %%% In case the entire image has been cropped away, we store a single
    %%% zero pixel for the variable.
    if isempty(SecondaryObjectImage)
        SecondaryObjectImage = 0;
    end
end
 
if any(size(SecondaryObjectImage) ~= size(PrimaryObjectImage))
    error(['Image processing was canceled in the ',ModuleName,' module due to an error in aligning the two object types'' images. They are not the same size.'])
end

ErodedPrimaryObjectImage = imerode(PrimaryObjectImage, ones(3));

SubregionObjectImage = SecondaryObjectImage;
SubregionObjectImage(ErodedPrimaryObjectImage~=0) = 0;

%%% Calculates object outlines
MaxFilteredImage = ordfilt2(SubregionObjectImage,9,ones(3,3),'symmetric');
%%% Determines the outlines.
IntensityOutlines = SubregionObjectImage - MaxFilteredImage;
%%% Converts to logical.
warning off MATLAB:conversionToLogical
FinalOutline = logical(IntensityOutlines);
warning on MATLAB:conversionToLogical

if ~isfield(handles.Measurements,SubregionObjectName)
    handles.Measurements.(SubregionObjectName) = {};
end

[handles,ChildList,FinalParentList] = CPrelateobjects(handles,SubregionObjectName,SecondaryObjectName,SubregionObjectImage,SecondaryObjectImage,ModuleName);
[handles,ChildList,FinalParentList] = CPrelateobjects(handles,SubregionObjectName,PrimaryObjectName,SubregionObjectImage,PrimaryObjectImage,ModuleName);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber);
    ColoredLabelMatrixImage = CPlabel2rgb(handles,SubregionObjectImage);
    SecondaryObjectImage = CPlabel2rgb(handles,SecondaryObjectImage);
    PrimaryObjectImage = CPlabel2rgb(handles,PrimaryObjectImage);

    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(PrimaryObjectImage,'TwoByTwo',ThisModuleFigureNumber);
    end
    subplot(2,2,1); 
    CPimagesc(PrimaryObjectImage,handles); 
    title([PrimaryObjectName, ' Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    subplot(2,2,2); 
    CPimagesc(SecondaryObjectImage,handles); 
    title([SecondaryObjectName, ' Image']);
    subplot(2,2,3); 
    CPimagesc(ColoredLabelMatrixImage,handles); 
    title([SubregionObjectName, ' Image']);
    subplot(2,2,4); 
    CPimagesc(FinalOutline,handles); 
    title([SubregionObjectName, ' Outlines']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the final, segmented label matrix image of secondary objects to
%%% the handles structure so it can be used by subsequent modules.
fieldname = ['Segmented', SubregionObjectName];
handles.Pipeline.(fieldname) = SubregionObjectImage;

handles = CPsaveObjectCount(handles, SubregionObjectName, SubregionObjectImage);
handles = CPsaveObjectLocations(handles, SubregionObjectName, SubregionObjectImage);

if ~strcmpi(SaveOutlines,'Do not save')
    handles.Pipeline.(SaveOutlines) = FinalOutline;
end