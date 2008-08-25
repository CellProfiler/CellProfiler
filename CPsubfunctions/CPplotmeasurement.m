function CPplotmeasurement(handles,PlotType,FigHandle,ModuleFlag,Object,Feature,Object2,Feature2)

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003--2008.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%% IF YOU CHANGE THIS SUBFUNCTION, BE SURE TO CONFIRM FUNCTIONALITY WITH
%%% ALL MODULES THAT CALL IT!

try FontSize = handles.Preferences.FontSize;
    %%% We used to store the font size in Current, so this line makes old
    %%% output files compatible. Shouldn't be necessary with any files made
    %%% after November 15th, 2006.
catch
    FontSize = handles.Current.FontSize;
end

str = pretty_feature_name(Feature, Object);

if PlotType <= 3
    UserAnswers = UserAnswersWindow(handles);
    if ~isfield(UserAnswers, 'FirstSample')
        return % Cancelled.
    end

    Measurements = handles.Measurements.(Object).(Feature);

    % Thresholds the data if user chose other than "None".  Reassigns
    % Measurements.
    if ~strcmp(UserAnswers.Logical,'None')
	msgfig=CPmsgbox('In the following dialog, please select the measurements that the original measurements will be thresholded on.');
	uiwait(msgfig);
	[object_name, feature_name] = CPgetfeature(handles, 1);
	if isempty(object_name)
	    return
	end
	[Thresholdstr, Measurements, object_name, feature_name] = ...
	    Threshold(handles, Measurements, UserAnswers, object_name, feature_name);
    end

    image_numbers = UserAnswers.FirstSample:UserAnswers.LastSample;
    nimages = length(image_numbers);

    empty_image = false;
    for image_number = image_numbers
        if isempty(Measurements{image_number})
            empty_image = true;
        end
    end
    if empty_image
        warnfig=CPwarndlg('There is an empty matrix in your measurement data, so a portion of the measurements will not be taken into account for the graph. This may affect the display of the graph (eg. fewer/no data points). This probably occurred because your custom-chosen data threshold was too stringent. You may consider trying a more lenient threshold.');
        uiwait(warnfig);
    end
    
    xticklabels = num2cell([0; image_numbers'; 0]);
    xticklabels{1} = '';
    xticklabels{end} = '';
end

if PlotType <= 2
    means = zeros(nimages, 1);
    stds = zeros(nimages, 1);

    for i = 1:nimages
	image_number = image_numbers(i);
	if ~isempty(Measurements{image_number})
	    means(i) = mean(Measurements{image_number});
	    stds(i) = std(Measurements{image_number});
	end
    end
end

if PlotType == 1
    % Bar graph %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if isempty(FigHandle)
        FigHandle=CPfigure;
    end

    graph = bar(means);
    set(graph, 'FaceColor', LineColor(UserAnswers.Color));

    hold on
    for k = 1:length(image_numbers)
        plot([k k],[means(k)-stds(k),means(k)+stds(k)],'k','linewidth',1)
        plot([k-0.075,k+0.075],[means(k)-stds(k),means(k)-stds(k)],'k','linewidth',1)
        plot([k-0.075,k+0.075],[means(k)+stds(k),means(k)+stds(k)],'k','linewidth',1)
    end
    hold off

    xlabel(gca,'Image number','Fontname','Helvetica','fontsize',FontSize+2)
    if strcmp(UserAnswers.Logical,'None') ~= 1
        ylabel(gca,{[];str;'for objects where';Thresholdstr;'mean +/- standard deviation'},'fontname','Helvetica','fontsize',FontSize+2)
    else
        ylabel(gca,[str,', mean +/- standard deviation'],'fontname','Helvetica','fontsize',FontSize+2)
    end
    axis([0 length(image_numbers)+1 ylim])
    set(gca,'xtick',[0 1:length(image_numbers) length(image_numbers)+1]);
    set(gca,'xticklabel', xticklabels);
    titlestr = str;

elseif PlotType == 2
    % Line graph %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if isempty(FigHandle)
        FigHandle=CPfigure;
    end
    hold on
    plot(1:nimages, means, 'Color', LineColor(UserAnswers.Color), 'LineWidth', 1);

    %%% Plots the Standard deviations as lines, too.
    plot(1:nimages, means-stds, ':', 'Color', LineColor(UserAnswers.Color));
    plot(1:nimages, means+stds, ':', 'Color', LineColor(UserAnswers.Color));
    hold off
    axis([0 nimages+1 ylim])

    xlabel(gca,'Image number','Fontname','Helvetica','fontsize',FontSize+2)
    if strcmp(UserAnswers.Logical,'None') ~= 1
        ylabel(gca,{[];str;'for objects where';Thresholdstr;'mean +/- standard deviation'},'fontname','Helvetica','fontsize',FontSize+2)
    else
        ylabel(gca,[str,', mean +/- standard deviation'],'fontname','Helvetica','fontsize',FontSize+2)
    end

    set(gca,'xtick',[0 1:nimages nimages+1]);
    set(gca,'xticklabel', xticklabels);
    titlestr = str;

elseif PlotType == 3
    % Scatter plot, 1 measurement %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if isempty(FigHandle)
        FigHandle=CPfigure;
    end

    hold on
    for k = UserAnswers.FirstSample:UserAnswers.LastSample
        if ~isempty(Measurements{k})
            plot(k*ones(length(Measurements{k})), Measurements{k}, '.k', 'color', LineColor(UserAnswers.Color))
            plot(k,mean(Measurements{k}), '.r', 'Markersize', 20)
        end
    end
    hold off
    axis([0 nimages+1 ylim])

    xlabel(gca,'Image number','Fontname','Helvetica','fontsize',FontSize+2)
    if strcmp(UserAnswers.Logical,'None') ~= 1
        ylabel(gca,{[];str;'for objects where';Thresholdstr},'fontname','Helvetica','fontsize',FontSize+2)
    else
        ylabel(gca,str,'fontname','Helvetica','fontsize',FontSize+2)
    end
    set(gca,'xtick',[0 1:nimages nimages+1]);
    set(gca,'xticklabel', xticklabels);
    titlestr = str;

elseif PlotType == 4
    % Scatter plot, 2 measurements %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Measurements1 = handles.Measurements.(Object).(Feature);
    Measurements2 = handles.Measurements.(Object2).(Feature2);

    if ModuleFlag == 0
        UserAnswers = UserAnswersWindow(handles);
        if ~isfield(UserAnswers, 'FirstSample')
            return % Cancelled.
        end

        %%% Thresholds the data if user chose other than "None"
        %%% Reassigns Measurements
        if ~strcmp(UserAnswers.Logical,'None')
	    msgfig=CPmsgbox('In the following dialog, please select the measurements that the original measurements will be thresholded on.');
	    uiwait(msgfig);
	    [object_name, feature_name] = CPgetfeature(handles, 1);
	    if isempty(object_name)
		return
	    end
	    [Thresholdstr2,Measurements2,object_name,feature_name] = Threshold(handles,Measurements2,UserAnswers,object_name,feature_name);
	    [Thresholdstr1,Measurements1,object_name,feature_name] = Threshold(handles,Measurements1,UserAnswers,object_name,feature_name);
        end

        if isempty(FigHandle)
            FigHandle=CPfigure;
        end

        hold on
        empty_image = false;
        for k = UserAnswers.FirstSample:UserAnswers.LastSample
            if size(Measurements1{k},1) ~= size(Measurements2{k})
                error('The number of objects for the chosen measurements does not match.')
            end
            if ~isempty(Measurements1{k}) && ~isempty(Measurements2{k})
                plot(Measurements1{k},Measurements2{k},'.k', 'color', LineColor(UserAnswers.Color))
            else
                empty_image = true;
            end
        end
        hold off

    else
        if isempty(FigHandle)
            FigHandle=CPfigure;
        end
        hold on
        empty_image = false;
        for k = 1:length(Measurements1)
            if size(Measurements1{k},1) ~= size(Measurements2{k})
                error('The number of objects for the chosen measurements does not match.')
            end
            if ~isempty(Measurements1{k})
                plot(Measurements1{k}, Measurements2{k}, '.k')
            else
                empty_image = true;
            end
        end
        hold off
    end

    if empty_image
        warnfig=CPwarndlg('There is an empty matrix in your measurement data, so a portion of the measurements will not be taken into account for the graph. This may affect the display of the graph (eg. fewer/no bars). This probably occurred because your custom-chosen data threshold was too stringent. You may consider trying a more lenient threshold.');
        uiwait(warnfig);
    end

    str2 = pretty_feature_name(Feature2, Object2);

    if ModuleFlag == 0 && strcmp(UserAnswers.Logical,'None') ~= 1
        xlabel(gca,{str;'for objects where';Thresholdstr1;[]},'fontsize',FontSize+2,'fontname','Helvetica')
        ylabel(gca,{[];str2;'for objects where';Thresholdstr2},'fontname','Helvetica','fontsize',FontSize+2)
    else
        xlabel(gca,str,'fontsize',FontSize+2,'fontname','Helvetica')
        ylabel(gca,str2,'fontname','Helvetica','fontsize',FontSize+2)
    end
    titlestr = [str2,' vs. ',str];
end

% Set some general figure and axes properties
set(gca,'fontname','Helvetica','fontsize',FontSize)
title(titlestr,'Fontname','Helvetica','fontsize',FontSize+2)
if ishandle(FigHandle)
    set(FigHandle,'name',[get(get(gca,'title'),'string'),': ',get(FigHandle,'name')])
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function UserAnswers = UserAnswersWindow(handles)
% This function displays a window for user input. If the return variable 'UserAnswers' is empty
% it means that either no measurements were found or the user pressed
% the Cancel button (or the window was closed).


% Store font size
FontSize = handles.Preferences.FontSize;

% Create UserWindow window
UserWindow = figure;
set(UserWindow,'units','inches','resize','on','menubar','none','toolbar','none','numbertitle','off','Name','Choose settings','Color',[.7 .7 .9]);
% Some variables controling the sizes of uicontrols
uiheight = 0.3;
% Set window size in inches, depends on the number of prompts
pos = get(UserWindow,'position');
Height = uiheight*10;
Width  = 5.8;
set(UserWindow,'position',[pos(1)+1 pos(2) Width Height]);

ypos = Height - uiheight*2.5;

NumMat=[];
for x=1:handles.Current.NumberOfImageSets
    NumMat=[NumMat;x];
end

ReverseNumMat=NumMat(end:-1:1);

% UserWindow user input
uicontrol(UserWindow,'style','text','String','First sample number to show or export:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','inches','position',[0.2 ypos 3.4 uiheight],'BackgroundColor',get(UserWindow,'color'));
FirstSample = uicontrol(UserWindow,'style','popupmenu','String',{NumMat},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[3.6 ypos+.05 1.8 uiheight],'BackgroundColor',get(UserWindow, 'color'));

ypos = ypos - uiheight;

uicontrol(UserWindow,'style','text','String','Last sample number to show or export:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','inches','position',[0.2 ypos 3.4 uiheight],'BackgroundColor',get(UserWindow,'color'));
LastSample = uicontrol(UserWindow,'style','popupmenu','String',{ReverseNumMat},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[3.6 ypos+.05 1.8 uiheight],'BackgroundColor',get(UserWindow, 'color'));

%Help button
LastSample_Help_Callback = 'CPhelpdlg(''To display data from only one image, choose the image number as both the first and last sample number.'')';
uicontrol(UserWindow,'style','pushbutton','String','?','FontSize',FontSize,...
    'HorizontalAlignment','center','units','inches','position',[5.45 ypos+.1 0.14 uiheight-0.05],'BackgroundColor',get(UserWindow,'color'),'FontWeight', 'bold',...
    'Callback', LastSample_Help_Callback);

ypos = ypos - uiheight;

uicontrol(UserWindow,'style','text','String','Color of the initial plot:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','inches','position',[0.2 ypos 3.4 uiheight],'BackgroundColor',get(UserWindow,'color'));
Color = uicontrol(UserWindow,'style','popupmenu','String',{'Blue','Red','Green','Yellow','Magenta','Cyan','Black','White','CellProfiler background'},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[3.6 ypos+.05 1.8 uiheight],'BackgroundColor',get(UserWindow, 'color'));

ypos = ypos - uiheight*2;

uicontrol(UserWindow,'style','text','String','Threshold applied to data:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','inches','position',[0.2 ypos 1.2 uiheight*1.5],'BackgroundColor',get(UserWindow,'color'));
Logical = uicontrol(UserWindow,'style','popupmenu','String',{'None','>','>=','=','<=','<'},'FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[1.6 ypos+.05 1.7 uiheight],'BackgroundColor',get(UserWindow, 'color'));

uicontrol(UserWindow,'style','text','String','If other than none, enter threshold value:','FontSize',FontSize,'FontWeight', 'bold',...
    'HorizontalAlignment','left','units','inches','position',[3.5 ypos 1.2 uiheight*1.5],'BackgroundColor',get(UserWindow,'color'));
ThresholdVal = uicontrol(UserWindow,'style','edit','String','','FontSize',FontSize,...
    'HorizontalAlignment','center','units','inches','position',[4.9 ypos+.05 .5 uiheight],'BackgroundColor',[1 1 1]);

%Help button
ThresholdVal_Help_Callback = 'CPhelpdlg(''Use this option if you want to calculate data only for objects meeting a threshold in a measurement.'')';
uicontrol(UserWindow,'style','pushbutton','String','?','FontSize',FontSize,...
    'HorizontalAlignment','center','units','inches','position',[5.45 ypos+.1 0.14 uiheight-0.05],'BackgroundColor',get(UserWindow,'color'),'FontWeight', 'bold',...
    'Callback', ThresholdVal_Help_Callback);


%%% OK AND CANCEL BUTTONS
posx = (Width - 1.7)/2;               % Centers buttons horizontally
okbutton = uicontrol(UserWindow,'style','pushbutton','String','OK',...
    'Fontweight','bold','FontSize',FontSize,'units','inches',...
    'position',[posx 0.1 0.75 0.3],'BackgroundColor',[.7 .7 .9],...
    'Callback','[cobj,cfig] = gcbo;set(cobj,''UserData'',1);uiresume(cfig);clear cobj cfig;',...
    'BackgroundColor',[.7 .7 .9]);
cancelbutton = uicontrol(UserWindow,'style','pushbutton','String','Cancel',...
    'Fontweight','bold','FontSize',FontSize,'units','inches',...
    'position',[posx+0.95 0.1 0.75 0.3],'Callback','close(gcf)','BackgroundColor',[.7 .7 .9]);


% Repeat until valid input has been entered or the window is destroyed
while 1

    % Wait until window is destroyed or uiresume() is called
    uiwait(UserWindow)

    % Action depending on user input
    if ishandle(okbutton)               % The OK button pressed
        %UserAnswers = get(UserWindow,'UserData');

        % Populate structure array
        UserAnswers.FirstSample = get(FirstSample,'value');
        UserAnswers.LastSample = ReverseNumMat(get(LastSample,'value'));
        UserAnswers.Color = get(Color,'value');
        UserAnswers.Logical = get(Logical,'value');
        UserAnswers.ThresholdVal = str2num(get(ThresholdVal,'string'));

        if UserAnswers.FirstSample > UserAnswers.LastSample         % Error check for sample numbers
            warnfig=CPwarndlg('Please make the first sample number less than or equal to the last sample number! Please try again.');
            uiwait(warnfig);
            set(okbutton,'UserData',[]);
        elseif UserAnswers.Logical ~= 1 & isempty(UserAnswers.ThresholdVal)                     % Error check for thresholding histogram data
            warnfig=CPwarndlg('You chose an option other than "None" for "Threshold applied to histogram data" but did not enter a valid threshold number. Please try again.');
            uiwait(warnfig);
            set(okbutton,'UserData',[]);
        else
            switch UserAnswers.Logical
                case 1
                    UserAnswers.Logical='None';
                case 2
                    UserAnswers.Logical='>';
                case 3
                    UserAnswers.Logical='>=';
                case 4
                    UserAnswers.Logical='=';
                case 5
                    UserAnswers.Logical='<=';
                otherwise
                    UserAnswers.Logical='<';
            end
            switch UserAnswers.Color
                case 1
                    UserAnswers.Color='Blue';
                case 2
                    UserAnswers.Color='Gray';
                case 3
                    UserAnswers.Color='Green';
                case 4
                    UserAnswers.Color='Yellow';
                case 5
                    UserAnswers.Color='Magenta';
                case 6
                    UserAnswers.Color='Cyan';
                case 7
                    UserAnswers.Color='Black';
                case 8
                    UserAnswers.Color='White';
                otherwise
                    UserAnswers.Color='CellProfiler background';
            end
            delete(UserWindow);
            return
        end
    else
        UserAnswers = [];
        if ishandle(UserWindow),delete(UserWindow);end
        return
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Thresholdstr,Measurements,ObjectTypename,FeatureType] = Threshold(handles,Measurements,UserAnswers,ObjectTypename,FeatureType)


if isempty(ObjectTypename)
    return
end
MeasurementToThresholdValueOnName = pretty_feature_name(FeatureType, ObjectTypename);
MeasurementToThresholdValueOn = handles.Measurements.(ObjectTypename).(FeatureType);

ArraySize=size(Measurements{1,1});
NumFeatures=ArraySize(2);
MeasureLen=ArraySize(1);

%%% Calculates the default bin size and range based on all the data.
% SelectedMeasurementsCellArray = Measurements(FirstSample:LastSample);
% SelectedMeasurementsMatrix = cell2mat(SelectedMeasurementsCellArray(:));

% Define variables
NumberOfImages=handles.Current.NumberOfImageSets;

Thresholdstr= [MeasurementToThresholdValueOnName,' ', UserAnswers.Logical,' ',num2str(UserAnswers.ThresholdVal)];
% AdditionalInfoForTitle = [' for objects where ', MeasurementToThresholdValueOnName,' ', UserAnswers.Logical,' ',num2str(UserAnswers.ThresholdVal)];

CompressedImageNumber = 1;
OutputMeasurements = cell(size(NumberOfImages,1),1);
% FinalHistogramData = [];
for ImageNumber = 1:NumberOfImages
    ListOfMeasurements{CompressedImageNumber,1} = Measurements{ImageNumber};
    ListOfMeasurements{CompressedImageNumber,2} = MeasurementToThresholdValueOn{ImageNumber};

    %%% Applies the specified ThresholdValue and gives a cell
    %%% array as output.
    operators = { @gt, @ge, @lt, @le, @eq };
    operator = operators{find(strcmp(UserAnswers.Logical, ...
				     { '>', '>=', '<', '<=' }))};
    newmat=[];
    boolcol = operator(ListOfMeasurements{CompressedImageNumber,2}, ...
		       UserAnswers.ThresholdVal);
    for col=1:NumFeatures
	datacol=ListOfMeasurements{CompressedImageNumber,1}(:,col);
	newcol=[];
	for row=1:length(boolcol)
	    if boolcol(row) == 1
		newcol=[newcol; datacol(row)];
	    end
	end
	
	newmat=[newmat newcol];
    end
    OutputMeasurements{CompressedImageNumber,1} = newmat;

    %%% Increments the CompressedImageNumber.
    CompressedImageNumber = CompressedImageNumber + 1;
end

Measurements=OutputMeasurements';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function lc = LineColor(color)
switch color
 case 'Blue'
  lc = 'b';
 case 'Gray'
  lc = [.7 .7 .7];
 case 'Green'
  lc = 'g';
 case 'Yellow'
  lc = 'y';
 case 'Magenta'
  lc = 'm';
 case 'Cyan'
  lc = 'c';
 case 'Black'
  lc = 'k';
 case 'White'
  lc = 'w';
 otherwise
  lc = [.7 .7 .9];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function str = pretty_feature_name(feature_name, object_name)
[feature_type, feature_subtype, image_name] = ...
    strread(feature_name, '%s%s%s', 'delimiter', '_');
if strcmp(feature_type, 'Intensity') ...
        || strcmp(feature_type, 'Texture') ...
        || strcmp(feature_type, 'Correlation') 
    str = sprintf('%s of %s in %s', char(feature_subtype), char(image_name), char(object_name));
else
    str = sprintf('%s of %s', strrep(char(feature_name), '_', ' '), char(object_name));
end
