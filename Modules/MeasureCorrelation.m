function handles = MeasureCorrelation(handles)

% Help for the Measure Correlation module:
% Category: Measurement
%
% Given two or more images, calculates the correlation between the
% pixel intensities. The correlation can be measured for the entire
% images, or individual correlation measurements can be made for each
% individual object, as defined by another module.
%
% See also MEASUREAREAOCCUPIED,
% MEASUREAREASHAPECOUNTLOCATION,
% MEASUREINTENSITYTEXTURE,
% MEASURETOTALINTENSITY.

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



%%% Reads the current module number, because this is needed to find the
%%% variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = Enter the names of each image type to be compared.
%choiceVAR01 = Do not use
%infotypeVAR01 = imagegroup
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = All pairwise comparisons will be performed.
%choiceVAR02 = Do not use
%infotypeVAR02 = imagegroup
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 =
%choiceVAR03 = Do not use
%infotypeVAR03 = imagegroup
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 =
%choiceVAR04 = Do not use
%infotypeVAR04 = imagegroup
ImageName{4} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = What did you call the objects within which to compare the images?
%choiceVAR05 = Do not use
%infotypeVAR05 = objectgroup
ObjectName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 =
%choiceVAR06 = Do not use
%infotypeVAR06 = objectgroup
ObjectName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 =
%choiceVAR07 = Do not use
%infotypeVAR07 = objectgroup
ObjectName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 =
%choiceVAR08 = Do not use
%infotypeVAR08 = objectgroup
ObjectName{4} = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Get the images
ImageCount = 0;
for ImageNbr = 1:4
    if ~strcmp(ImageName{ImageNbr},'Do not use')
        ImageCount = ImageCount + 1;
        try

            %%% Checks whether image has been loaded.
            if ~isfield(handles.Pipeline,ImageName{ImageNbr})
                %%% If the image is not there, an error message is produced.  The error
                %%% is not displayed: The error function halts the current function and
                %%% returns control to the calling function (the analyze all images
                %%% button callback.)  That callback recognizes that an error was
                %%% produced because of its try/catch loop and breaks out of the image
                %%% analysis loop without attempting further modules.
                error(['Image processing was canceled because the Measure Correlation module could not find the input image.  It was supposed to be named ', ImageName{ImageNbr}, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
            end
            Image{ImageCount} = handles.Pipeline.(ImageName{ImageNbr});
            tmpImageName{ImageCount} = ImageName{ImageNbr};
            %%% Checks that the original image is two-dimensional (i.e. not a color
            %%% image), which would disrupt several of the image functions.
            if ndims(Image{ImageCount}) ~= 2
                error('Image processing was canceled because the Measure Correlation module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
            end

        catch error(['There was a problem loading the image you called ', ImageName{ImageNbr}, ' in the Measure Correlation module.'])
        end
    end
end
ImageName = tmpImageName;           % Get rid of '/' in the ImageName cell array so we don't have to care about them later.

% Check so that at least two images have been entered
if ImageCount < 2
    errordlg('At least two image names must be entered in the MeasureCorrelation module.')
end

%%% Get the masks of segemented objects
ObjectNameCount = 0;
for ObjectNameNbr = 1:4
    if ~strcmp(ObjectName{ObjectNameNbr},'Do not use')
        ObjectNameCount = ObjectNameCount + 1;
        tmpObjectName{ObjectNameCount} = ObjectName{ObjectNameNbr};
        if ~strcmp(ObjectName{ObjectNameNbr},'Image')
            %%% Retrieves the label matrix image that contains the
            %%% segmented objects which will be used as a mask. Checks first to see
            %%% whether the appropriate image exists.
            fieldname = ['Segmented', ObjectName{ObjectNameNbr}];
            %%% Checks whether the image exists in the handles structure.
            if isfield(handles.Pipeline, fieldname)==0,
                error(['Image processing has been canceled. Prior to running the Measure Correlation module, you must have previously run a module that generates an image with the primary objects identified.  You specified in the Measure Correlation module that the objects were named ', ObjectName, ' as a result of a previous module, which should have produced an image called ', fieldname, ' in the handles structure.  The Measure Correlation module cannot locate this image.']);
            end
            LabelMatrixImage{ObjectNameCount} = handles.Pipeline.(fieldname);
        else
            LabelMatrixImage{ObjectNameCount} = ones(size(Image{1}));        % Use mask of ones to indicate that the correlation should be calcualted for the entire image
        end
    end
end
ObjectName = tmpObjectName; % Get rid of '/' in the ObjectName cell array so we don't have to care about them later.

% Check so that at least one object type have been entered
if ObjectNameCount < 1
    errordlg('At least one object type must be entered in the MeasureCorrelation module.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow




% Produce feature names for all pairwise image combinations
CorrelationFeatures = {};
for i = 1:ImageCount-1
    for j = i+1:ImageCount
        CorrelationFeatures{end+1} = ['Correlation ',ImageName{i},' and ',ImageName{j}];
    end
end

% For each object type and for each segmented object, calculate the correlation between all combinations of images
for ObjectNameNbr = 1:ObjectNameCount

    % Calculate the correlation in all objects for all pairwise image combinations
    NbrOfObjects = max(LabelMatrixImage{ObjectNameNbr}(:));          % Get number of segmented objects
    Correlation = zeros(NbrOfObjects,length(CorrelationFeatures));   % Pre-allocate memory
    for ObjectNbr = 1:NbrOfObjects                                   % Loop over objects
        FeatureNbr = 1;                                              % Easiest way to keep track of the feature number, i.e. which combination of images
        for i = 1:ImageCount-1                                       % Loop over all combinations of images
            for j = i+1:ImageCount
                index = find(LabelMatrixImage{ObjectNameNbr} == ObjectNbr);   % Get the indexes for the this object number
                c = corrcoef([Image{i}(index) Image{j}(index)]);              % Get the values for these indexes in the images and calculate the correlation
                Correlation(ObjectNbr,FeatureNbr) = c(1,2);                   % Store the correlation
                FeatureNbr = FeatureNbr + 1;
            end
        end
    end

    % Store the correlation measurements
    handles.Measurements.(ObjectName{ObjectNameNbr}).CorrelationFeatures = CorrelationFeatures;
    handles.Measurements.(ObjectName{ObjectNameNbr}).Correlation(handles.Current.SetBeingAnalyzed) = {Correlation};
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

    % Set white background color
    set(ThisModuleFigureNumber,'Color',[1 1 1])

    % Get size of window
    Position = get(ThisModuleFigureNumber,'Position');
    Height = Position(4);
    Width  = Position(3);

    delete(findobj('Parent',ThisModuleFigureNumber));

    %%% Displays the results.
    Displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text', 'position', [0 Height-40 Width 20],...
        'fontname','times','FontSize',handles.Current.FontSize,'backgroundcolor',[1,1,1],'horizontalalignment','center','fontweight','bold');
    TextToDisplay = ['Average correlations in Image Set # ',num2str(handles.Current.SetBeingAnalyzed)];
    set(Displaytexthandle,'string',TextToDisplay)


    for ObjectNameNbr = 0:ObjectNameCount
        row = 1;

        % Write object names
        if ObjectNameNbr > 0         % Don't write any object type name in the first colum
            h = uicontrol(ThisModuleFigureNumber,'style','text','position',[110+70*ObjectNameNbr Height-110 70 25],...
                'fontname','times','FontSize',handles.Current.FontSize,'backgroundcolor',[1,1,1],'horizontalalignment','center',...
                'fontweight','bold');
            set(h,'string',ObjectName{ObjectNameNbr});
        end

        % Write image names or correlation measurements
        FeatureNbr = 1;
        for i = 1:ImageCount-1
            for j = i+1:ImageCount
                if ObjectNameNbr == 0               % First column, write image names
                    h = uicontrol(ThisModuleFigureNumber,'style','text','position',[20 Height-120-40*row 120 40],...
                        'fontname','times','FontSize',handles.Current.FontSize,'backgroundcolor',[1,1,1],'horizontalalignment','left',...
                        'fontweight','bold');
                    TextToDisplay = sprintf('%s and \n%s',ImageName{i},ImageName{j});
                    set(h,'string',TextToDisplay);
                else
                    % Calculate the average correlation over the objects
                    c = mean(handles.Measurements.(ObjectName{ObjectNameNbr}).Correlation{handles.Current.SetBeingAnalyzed}(:,FeatureNbr));
                    uicontrol(ThisModuleFigureNumber,'style','text','position',[110+70*ObjectNameNbr Height-125-40*row 70 40],...
                        'fontname','times','FontSize',handles.Current.FontSize,'backgroundcolor',[1,1,1],'horizontalalignment','center',...
                        'string',sprintf('%0.2f',c));
                    FeatureNbr = FeatureNbr + 1;
                end
                row = row + 1;
            end
        end
    end
    set(ThisModuleFigureNumber,'toolbar','figure')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow


