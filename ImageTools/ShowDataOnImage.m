function handles = ShowDataOnImage(handles)

% Help for the Show Data on Image tool:
% Category: Image Tools
%
% This allows you to extract measurements from an output file and
% overlay any measurements that you have made on any image. For
% example, you could look at the DNA content (e.g.
% IntegratedIntensityOrigBlue) of each cell on an image of nuclei.
% Or, you could look at cell area on an image of nuclei.  
% 
% First, you are asked to select the measurement you want to be
% displayed on the image.  Next, you are asked to select the X and
% then the Y locations where these measurements should be displayed.
% Typically, your options are the XY locations of the nuclei, or the
% XY locations of the cells, and these are usually named something
% like 'CenterXNuclei'.  If your output file has measurements from
% many images, you then select which sample number to view.
% 
% Then, CellProfilerTM tries to guide you to find the image that
% corresponds to this sample number.  First, it asks which file name
% would be most helpful for you to find the image. CellProfilerTM
% uses whatever you enter here to look up the exact file name you are
% looking for, so that you can browse to find the image. Once the
% image is selected, extraction ensues and eventually the image will
% be shown with the measurements on top.
% 
% You can use the tools at the top to zoom in on this image. If the
% text is overlapping and not easily visible, you can change the
% number of decimal places shown with the 'Fewer significant digits'
% button, or you can change the font size with the 'Text Properties'.
% You can also change the font style, color, and other properties with
% this button.  
% 
% The resulting figure can be saved in Matlab format (.fig) or
% exported in a traditional image file format.
%
% See also SHOWIMAGE, SHOWPIXELDATA.

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

cd(handles.Current.DefaultOutputDirectory)
%%% Asks the user to choose the file from which to extract measurements.
[RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');
if RawFileName ~= 0
    load(fullfile(RawPathname,RawFileName));
    %%% Extracts the fieldnames of measurements from the handles structure. 
    Fieldnames = fieldnames(handles.Measurements);
    MeasFieldnames = Fieldnames(strncmp(Fieldnames,'Object',6)==1);
    %%% Error detection.
    if isempty(MeasFieldnames)
        errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles.Measurements structure preceded by ''Object''.')
        cd(handles.Current.StartupDirectory);
        return
    else
        %%% Removes the 'Object' prefix from each name for display purposes.
        for Number = 1:length(MeasFieldnames)
            EditedMeasFieldnames{Number} = MeasFieldnames{Number}(7:end);
        end
        %%% Allows the user to select a measurement from the list.
        [Selection, ok] = listdlg('ListString',EditedMeasFieldnames, 'ListSize', [300 600],...
            'Name','Select measurement',...
            'PromptString','Choose a measurement to display on the image','CancelString','Cancel',...
            'SelectionMode','single');
        if ok ~= 0
            EditedMeasurementToExtract = char(EditedMeasFieldnames(Selection));
            MeasurementToExtract = ['Object', EditedMeasurementToExtract];
            %%% Allows the user to select the X Locations from the list.
            [Selection, ok] = listdlg('ListString',EditedMeasFieldnames, 'ListSize', [300 600],...
                'Name','Select the X locations to be used',...
                'PromptString','Select the X locations to be used','CancelString','Cancel',...
                'SelectionMode','single');
            if ok ~= 0
                EditedXLocationMeasurementName = char(EditedMeasFieldnames(Selection));
                XLocationMeasurementName = ['Object', EditedXLocationMeasurementName];
                %%% Allows the user to select the Y Locations from the list.
                [Selection, ok] = listdlg('ListString',EditedMeasFieldnames, 'ListSize', [300 600],...
                    'Name','Select the Y locations to be used',...
                    'PromptString','Select the Y locations to be used','CancelString','Cancel',...
                    'SelectionMode','single');
                if ok ~= 0
                    EditedYLocationMeasurementName = char(EditedMeasFieldnames(Selection));
                    YLocationMeasurementName = ['Object', EditedYLocationMeasurementName];
                    %%% Prompts the user to choose a sample number to be displayed.
                    Answer = inputdlg({'Which sample number do you want to display?'},'Choose sample number',1,{'1'});
                    if isempty(Answer)
                        cd(handles.Current.StartupDirectory);
                        return
                    end
                    SampleNumber = str2double(Answer{1});
                    TotalNumberImageSets = length(handles.Measurements.(MeasurementToExtract));
                    if SampleNumber > TotalNumberImageSets
                        cd(handles.Current.StartupDirectory);
                        error(['The number you entered exceeds the number of samples in the file.  You entered ', num2str(SampleNumber), ' but there are only ', num2str(TotalNumberImageSets), ' in the file.'])
                    end
                    %%% Looks up the corresponding image file name.
                    Fieldnames = fieldnames(handles.Measurements);
                    PotentialImageNames = Fieldnames(strncmp(Fieldnames,'Filename',8)==1);
                    %%% Error detection.
                    if isempty(PotentialImageNames)
                        errordlg('CellProfiler was not able to look up the image file names used to create these measurements to help you choose the correct image on which to display the results. You may continue, but you are on your own to choose the correct image file.')
                    end
                    %%% Allows the user to select a filename from the list.
                    [Selection, ok] = listdlg('ListString',PotentialImageNames, 'ListSize', [300 600],...
                        'Name','Choose the image whose filename you want to display',...
                        'PromptString','Choose the image whose filename you want to display','CancelString','Cancel',...
                        'SelectionMode','single');
                    if ok ~= 0
                        SelectedImageName = char(PotentialImageNames(Selection));
                        ImageFileName = handles.Measurements.(SelectedImageName){SampleNumber};
                        %%% Prompts the user with the image file name.
                        h = msgbox(['Browse to find the image called ', ImageFileName,'.']);
                        %%% Opens a user interface window which retrieves a file name and path 
                        %%% name for the image to be displayed.
                        cd(handles.Current.DefaultImageDirectory)
                        [FileName,Pathname] = uigetfile('*.*','Select the image to view');
                        delete(h)
                        %%% If the user presses "Cancel", the FileName will = 0 and nothing will
                        %%% happen.
                        if FileName == 0
                            cd(handles.Current.StartupDirectory);
                            return
                        else
                            %%% Opens and displays the image, with pixval shown.
                            ImageToDisplay = imcpread(fullfile(Pathname,FileName));
                            %%% Allows underscores to be displayed properly.
                            ImageFileName = strrep(ImageFileName,'_','\_');
                            FigureHandle = figure; imagesc(ImageToDisplay), colormap(gray), title([EditedMeasurementToExtract, ' on ', ImageFileName])
                            %%% Extracts the XY locations and the measurement values.
                            global StringListOfMeasurements
                            ListOfMeasurements = handles.Measurements.(MeasurementToExtract){SampleNumber};
                            StringListOfMeasurements = cellstr(num2str(ListOfMeasurements));
                            Xlocations(:,FigureHandle) = handles.Measurements.(XLocationMeasurementName){SampleNumber};
                            Ylocations(:,FigureHandle) = handles.Measurements.(YLocationMeasurementName){SampleNumber};
                            %%% A button is created in the display window which
                            %%% allows altering the properties of the text.
                            StdUnit = 'point';
                            StdColor = get(0,'DefaultUIcontrolBackgroundColor');
                            PointsPerPixel = 72/get(0,'ScreenPixelsPerInch');                            
                            DisplayButtonCallback1 = 'global TextHandles, FigureHandle = gcf; CurrentTextHandles = TextHandles{FigureHandle}; try, propedit(CurrentTextHandles,''v6''); catch, msgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; drawnow, clear TextHandles';
                            uicontrol('Parent',FigureHandle, ...
                                'Unit',StdUnit, ...
                                'BackgroundColor',StdColor, ...
                                'CallBack',DisplayButtonCallback1, ...
                                'Position',PointsPerPixel*[2 2 90 22], ...
                                'Units','Normalized',...
                                'String','Text Properties', ...
                                'Style','pushbutton');
                            DisplayButtonCallback2 = 'global TextHandles StringListOfMeasurements, FigureHandle = gcf; NumberOfDecimals = inputdlg(''Enter the number of decimal places to display'',''Enter the number of decimal places'',1,{''0''}); CurrentTextHandles = TextHandles{FigureHandle}; NumberValues = str2num(cell2mat(StringListOfMeasurements)); Command = [''%.'',num2str(NumberOfDecimals{1}),''f'']; NewNumberValues = num2str(NumberValues,Command); CellNumberValues = cellstr(NewNumberValues); PropName(1) = {''string''}; set(CurrentTextHandles,PropName, CellNumberValues); drawnow, clear TextHandles StringListOfMeasurements';
                            uicontrol('Parent',FigureHandle, ...
                                'Unit',StdUnit, ...
                                'BackgroundColor',StdColor, ...
                                'CallBack',DisplayButtonCallback2, ...
                                'Position',PointsPerPixel*[100 2 135 22], ...
                                'Units','Normalized',...
                                'String','Fewer significant digits', ...
                                'Style','pushbutton');
                            DisplayButtonCallback3 = 'global TextHandles StringListOfMeasurements, FigureHandle = gcf; CurrentTextHandles = TextHandles{FigureHandle}; PropName(1) = {''string''}; set(CurrentTextHandles,PropName, StringListOfMeasurements); drawnow, clear TextHandles StringListOfMeasurements';
                            uicontrol('Parent',FigureHandle, ...
                                'Unit',StdUnit, ...
                                'BackgroundColor',StdColor, ...
                                'CallBack',DisplayButtonCallback3, ...
                                'Position',PointsPerPixel*[240 2 135 22], ...
                                'Units','Normalized',...
                                'String','Restore labels', ...
                                'Style','pushbutton');
                            DisplayButtonCallback4 = 'global TextHandles StringListOfMeasurements, FigureHandle = gcf; CurrentTextHandles = TextHandles{FigureHandle}; set(CurrentTextHandles, ''visible'', ''off''); drawnow, clear TextHandles StringListOfMeasurements';
                            uicontrol('Parent',FigureHandle, ...
                                'Unit',StdUnit, ...
                                'BackgroundColor',StdColor, ...
                                'CallBack',DisplayButtonCallback4, ...
                                'Position',PointsPerPixel*[380 2 85 22], ...
                                'Units','Normalized',...
                                'String','Hide labels', ...
                                'Style','pushbutton');
                            DisplayButtonCallback5 = 'global TextHandles StringListOfMeasurements, FigureHandle = gcf; CurrentTextHandles = TextHandles{FigureHandle}; set(CurrentTextHandles, ''visible'', ''on''); drawnow, clear TextHandles StringListOfMeasurements';
                            uicontrol('Parent',FigureHandle, ...
                                'Unit',StdUnit, ...
                                'BackgroundColor',StdColor, ...
                                'CallBack',DisplayButtonCallback5, ...
                                'Position',PointsPerPixel*[470 2 85 22], ...
                                'Units','Normalized',...
                                'String','Show labels', ...
                                'Style','pushbutton');
                            %%% Overlays the values in the proper location in the
                            %%% image.
                            global TextHandles
                            TextHandles{FigureHandle} = text(Xlocations(:,FigureHandle) , Ylocations(:,FigureHandle) , StringListOfMeasurements,...
                                'HorizontalAlignment','center', 'color', 'white');
                            %%% Puts the menu and tool bar in the figure window.
                            set(FigureHandle,'toolbar', 'figure')
                        end
                    end
                end    
            end
        end
    end
end
cd(handles.Current.StartupDirectory);