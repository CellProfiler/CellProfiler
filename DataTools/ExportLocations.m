function ExportLocations(handles)

% Help for the Export Locations tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Exports center locations of objects. Specialty function for creating a
% locations list for microscopy image acquisition of gridded spots.
% *************************************************************************
%
% Useful for creating a locations list for microscope.

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

%%% Ask the user to choose the file from which to extract
%%% measurements. The window opens in the default output directory.
[RawFileName, RawPathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'.','*.mat'),'Select the raw measurements file');
%%% Allows canceling.
if RawFileName == 0
    return
end
load(fullfile(RawPathname, RawFileName));

if ~exist('handles','var')
    CPerrordlg('This is not a CellProfiler output file.');
    return
end

%%% Quick check if it seems to be a CellProfiler file or not
if ~isfield(handles,'Measurements')
    CPerrordlg('The selected file does not contain any measurements.')
    return
end

%%% Extract the fieldnames of measurements from the handles structure.
MeasFieldnames = fieldnames(handles.Measurements);
for i=1:length(handles.Measurements)
    if strcmp(MeasFieldnames{i},'Image')
        MeasFieldnames(i) = [];
    end
    if isempty(MeasFieldnames)
        CPerrordlg('The output file you have chosen does not have location measurements.');
        return
    end
end

[Selection, ok] = listdlg('ListString',MeasFieldnames,'ListSize', [300 200],...
    'Name','Select measurement',...
    'PromptString','Which object do you want to export locations from?',...
    'CancelString','Cancel',...
    'SelectionMode','single');

if ok == 0
    return
end

ObjectTypename = MeasFieldnames{Selection};

if isfield(handles.Measurements.(ObjectTypename),'Location')
    Locations = handles.Measurements.(ObjectTypename).Location;
else
    CPerrordlg('The object you have chosen does not have location measurements.');
    return
end

AcceptableAnswers = 0;
Prompts{1} = 'What do you want the x,y location of the first object to be (in final units)?';
Prompts{2} = 'How many units per pixel (e.g. microns)?';
Prompts{3} = 'If based on a grid, do you want locations in comb or meander format (ignore otherwise)?';
Prompts{4} = 'Do you want to correct for uneven sub-grids?';

Defaults{1} = '0,0';
Defaults{2} = '10';
Defaults{3} = 'Comb';
Defaults{4} = 'No';

while AcceptableAnswers == 0

    Answers = inputdlg(Prompts(1:4),'Export Locations Settings',1,Defaults(1:4),'on');

    FirstObjectLocation = Answers{1};
    comma = strfind(FirstObjectLocation,',');
    commaerrortext = 'The x,y location must be entered in the following format: 0,0';
    if length(comma) > 1
        uiwait(CPwarndlg(commaerrortext));
        continue
    else
        FirstSpotX = str2double(FirstObjectLocation(1:comma-1));
        FirstSpotY = str2double(FirstObjectLocation(comma+1:end));
        if isempty(FirstSpotX) || isempty(FirstSpotY)
            uiwait(CPwarndlg(commaerrortext));
            continue
        end
    end

    PixelUnits = str2double(Answers{2});
    if isempty(PixelUnits)
        uiwait(CPwarndlg('Units per pixel must be a valid number.'));
        continue
    end

    MeanderOption = Answers{3};
    if ~strcmpi(MeanderOption,'Comb') && ~strcmpi(MeanderOption,'Meander')
        uiwait(CPwarndlg('You did not enter comb or meander, so locations will be left as default.'));
    end

    GridChoice = Answers{4};
    if ~strcmpi(GridChoice,'Yes') && ~strcmpi(GridChoice,'No')
        uiwait(CPwarndlg('You must answer either yes or no for correcting uneven grids.'));
        continue
    end
    AcceptableAnswers = 1;
end

if strcmpi(MeanderOption,'Meander') || strcmpi(GridChoice,'Yes')

    Fields=fieldnames(handles.Measurements.Image);
    GridList={};
    for i = 1:length(Fields)
        if strcmp(Fields{i}(end-3:end),'Info')
            GridList{end+1}=Fields{i}; %#ok Ignore MLint
        end
    end

    if ~isempty(GridList)
        Selection = listdlg('ListString',GridList,'ListSize', [300 200],...
            'Name','Select Grid',...
            'PromptString','Select grid to base grid correction on',...
            'CancelString','Cancel',...
            'SelectionMode','single');
        GridToBaseCorrectionOn=handles.Measurements.Image.(GridList{Selection}){1};
        GridInfo.XLocationOfLowestXSpot = GridToBaseCorrectionOn(1);
        GridInfo.YLocationOfLowestYSpot = GridToBaseCorrectionOn(2);
        GridInfo.XSpacing = GridToBaseCorrectionOn(3);
        GridInfo.YSpacing = GridToBaseCorrectionOn(4);
        GridInfo.Rows = GridToBaseCorrectionOn(5);
        GridInfo.Columns = GridToBaseCorrectionOn(6);
        GridInfo.TotalHeight = GridToBaseCorrectionOn(7);
        GridInfo.TotalWidth = GridToBaseCorrectionOn(8);
        if GridToBaseCorrectionOn(9) == 1
            GridInfo.LeftOrRight = 'Left';
        else
            GridInfo.LeftOrRight = 'Right';
        end
        if GridToBaseCorrectionOn(10) == 1
            GridInfo.TopOrBottom = 'Top';
        else
            GridInfo.TopOrBottom = 'Bottom';
        end
        if GridToBaseCorrectionOn(11) == 1
            GridInfo.RowsOrColumns = 'Rows';
        else
            GridInfo.RowsOrColumns = 'Columns';
        end
        Grid = CPmakegrid(GridInfo);
        VertLinesX = Grid.VertLinesX;
        HorizLinesY = Grid.HorizLinesY;
        EntireGridRows = GridToBaseCorrectionOn(5);
        EntireGridCols = GridToBaseCorrectionOn(6);
    else
        error('Can''t do grid correction.');
    end

    if strcmpi(GridChoice,'Yes')
        try
            ObjectAreas = handles.Measurements.(ObjectTypename).AreaShape{1}(:,1);
        catch
            CPerrordlg('The object you have chosen does not have Area measurements.');
        end

        AcceptableAnswers = 0;
        while AcceptableAnswers == 0
            Prompts{1} = 'How many rows in the sub-grid?';
            Prompts{2} = 'How many columns in the sub-grid?';

            Defaults{1} = '10';
            Defaults{2} = '10';

            Answers = inputdlg(Prompts(1:2),'Export Locations Settings',1,Defaults(1:2),'on');

            SubGridRows = round(str2double(Answers{1}));
            SubGridCols = round(str2double(Answers{2}));

            if isempty(SubGridRows) || isempty(SubGridCols)
                uiwait(CPwarndlg('You must enter integers for all grid values.'));
                continue
            end

            if rem(EntireGridRows,SubGridRows) || rem(EntireGridCols,SubGridCols)
                uiwait(CPwarndlg('The entire grids rows and columns must be divisible by sub grids rows and columns.'));
                continue
            end

            AcceptableAnswers = 1;
        end

        OldXLocations = reshape(Locations{1}(:,1),EntireGridCols,EntireGridRows)';
        OldYLocations = reshape(Locations{1}(:,2),EntireGridCols,EntireGridRows)';
        OldObjectAreas = reshape(ObjectAreas,EntireGridCols,EntireGridRows)';

        for i = 1:(EntireGridRows/SubGridRows)
            for j = 1:(EntireGridCols/SubGridCols)
                SmallXLocations = OldXLocations((i*SubGridRows-SubGridRows+1):(i*SubGridRows),(j*SubGridCols-SubGridCols+1):(j*SubGridCols));
                SmallYLocations = OldYLocations((i*SubGridRows-SubGridRows+1):(i*SubGridRows),(j*SubGridCols-SubGridCols+1):(j*SubGridCols));
                SmallAreas = OldObjectAreas((i*SubGridRows-SubGridRows+1):(i*SubGridRows),(j*SubGridCols-SubGridCols+1):(j*SubGridCols));

                YGridLines=HorizLinesY(1,((i-1)*SubGridRows+1):(i*SubGridRows));
                XGridLines=VertLinesX(1,((j-1)*SubGridCols+1):(j*SubGridCols));
                NormedXVals=[];
                NormedYVals=[];
                for a = 1:SubGridRows
                    for b = 1:SubGridCols
                        Yval = SmallYLocations(a,b);
                        Xval = SmallXLocations(a,b);
                        SpotArea = SmallAreas(a,b);
                        if SpotArea > 1
                            NormedXVals(end+1)=Xval-XGridLines(b); %#ok Ignore MLint
                            NormedYVals(end+1)=Yval-YGridLines(a); %#ok Ignore MLint
                        end
                    end
                end

                DifferenceXVal=abs(NormedXVals-mean(NormedXVals));
                DifferenceYVal=abs(NormedYVals-mean(NormedYVals));
                SortedDifferenceXVal=sort(DifferenceXVal);
                SortedDifferenceYVal=sort(DifferenceYVal);
                NewNormedXVals=NormedXVals(find(DifferenceXVal<SortedDifferenceXVal((round(length(SortedDifferenceXVal)/2))))); %#ok Ignore MLint
                NewNormedYVals=NormedYVals(find(DifferenceYVal<SortedDifferenceYVal((round(length(SortedDifferenceYVal)/2))))); %#ok Ignore MLint
                DifferenceXVal=abs(NormedXVals-mean(NewNormedXVals));
                DifferenceYVal=abs(NormedYVals-mean(NewNormedYVals));
                SortedDifferenceXVal=sort(DifferenceXVal);
                SortedDifferenceYVal=sort(DifferenceYVal);
                NewNormedXVals=NormedXVals(find(DifferenceXVal<SortedDifferenceXVal((round(length(SortedDifferenceXVal)/2))))); %#ok Ignore MLint
                NewNormedYVals=NormedYVals(find(DifferenceYVal<SortedDifferenceYVal((round(length(SortedDifferenceYVal)/2))))); %#ok Ignore MLint

                FinalXVal=mean(NewNormedXVals);
                FinalYVal=mean(NewNormedYVals);

                for a = 1:SubGridRows
                    NewYvals(a,1:SubGridRows) = YGridLines(a)+FinalYVal; %#ok Ignore MLint
                end

                for b = 1:SubGridCols
                    NewXvals(1:SubGridCols,b) = XGridLines(b)+FinalXVal; %#ok Ignore MLint
                end

                NewXLocations((i*SubGridRows-SubGridRows+1):(i*SubGridRows),(j*SubGridCols-SubGridCols+1):(j*SubGridCols)) = NewXvals;
                NewYLocations((i*SubGridRows-SubGridRows+1):(i*SubGridRows),(j*SubGridCols-SubGridCols+1):(j*SubGridCols)) = NewYvals;
            end
        end

        NewLocations{1}(:,1) = reshape(NewXLocations',1,[]);
        NewLocations{1}(:,2) = reshape(NewYLocations',1,[]);

        Locations = NewLocations;
    end

    if strcmpi(MeanderOption,'Meander')
        for i = 2:2:EntireGridRows
            Locations{1}((EntireGridCols*(i-1)+1):EntireGridCols*(i-1)+EntireGridCols,:)=flipud(Locations{1}((EntireGridCols*(i-1)+1):EntireGridCols*(i-1)+EntireGridCols,:));
        end
    end
end

for ImageNumber = 1:length(Locations)
    filename = [ObjectTypename,'_Locations_Image_',num2str(ImageNumber),'.csv'];
    fid = fopen(fullfile(handles.Current.DefaultOutputDirectory,filename),'w');
    if fid == -1
        CPerrordlg(sprintf('Cannot create the output file %s. There might be another program using a file with the same name.',filename));
        return
    end
    FixedLocations = Locations{ImageNumber}*PixelUnits;
    FixedLocations(:,1) = FixedLocations(:,1) - (FixedLocations(1,1)-FirstSpotX);
    FixedLocations(:,2) = FixedLocations(:,2) - (FixedLocations(1,2)-FirstSpotY);
    for ObjectNumber = 1:size(Locations{ImageNumber},1)
        fprintf(fid,[num2str(round(FixedLocations(ObjectNumber,1))),',',num2str(round(FixedLocations(ObjectNumber,2))),'\n']);
    end
    fclose(fid);
end