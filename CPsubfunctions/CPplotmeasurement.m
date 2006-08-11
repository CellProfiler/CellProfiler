function CPplotmeasurement(handles,PlotType,ModuleFlag,Object,Feature,FeatureNo,Object2,Feature2,FeatureNo2)

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
% $Revision: 2802 $

try FontSize = handles.Preferences.FontSize;
    %%% We used to store the font size in Current, so this line makes old
    %%% output files compatible. Shouldn't be necessary with any files made
    %%% after November 15th, 2006.
catch
    FontSize = handles.Current.FontSize;
end

if (length(Feature) > 10) & strncmp(Feature,'Intensity_',10)
    str = [handles.Measurements.(Object).([Feature,'Features']){FeatureNo},' of ',Object,' in ',Feature(11:end)];
elseif (length(Feature) > 8) & strncmp(Feature,'Texture_',8)
    str = [handles.Measurements.(Object).([Feature,'Features']){FeatureNo},' of ',Object,' in ',Feature(11:end)];
else
    str = [handles.Measurements.(Object).([Feature,'Features']){FeatureNo},' of ',Object];
end

% Bar graph
if PlotType == 1
    %%% Extract the measurement
    Measurements = handles.Measurements.(Object).(Feature);

    %%% Opens a window that lets the user choose graph settings
    %%% This function returns a UserAnswers structure with the
    %%% information required to carry out the calculations.
    try UserAnswers = UserAnswersWindow(handles);
    catch CPerrordlg(lasterr)
        return
    end

    % If Cancel button pressed, return
    if ~isfield(UserAnswers, 'FirstSample')
        return
    end

    FirstImage=UserAnswers.FirstSample;
    LastImage=UserAnswers.LastSample;

    switch UserAnswers.Color
        case 'Blue'
            LineColor='b';
        case 'Gray'
            LineColor=[.7 .7 .7];
        case 'Green'
            LineColor='g';
        case 'Yellow'
            LineColor='y';
        case 'Magenta'
            LineColor='m';
        case 'Cyan'
            LineColor='c';
        case 'Black'
            LineColor='k';
        case 'White'
            LineColor='w';
        otherwise
            LineColor=[.7 .7 .9];
    end
    
    PlotNum=1;
    %%% Thresholds the data if user chose other than "None"
    %%% Reassigns Measurements
    if strcmp(UserAnswers.Logical,'None') ~= 1
        try
            msgfig=CPmsgbox('In the following dialog, please select the measurements that the original measurements will be thresholded on.');
            uiwait(msgfig);
            [ObjectTypename,FeatureType,FeatureNum] = CPgetfeature(handles);
            % There's no need to nest the call to CPgetfeature again, because
            % the only errors that can occur in CPgetfeature happen when
            % handles is faulty, but CPgetfeature was called before and handles
            % hasn't been modified. However, an empty check for ObjectTypename
            % is needed. This would happen if the user clicked Cancel.
            [Thresholdstr,Measurements,ObjectTypename,FeatureType,FeatureNum] = Threshold(handles,Measurements,UserAnswers,ObjectTypename,FeatureType,FeatureNum);
        catch CPerrordlg(lasterr)
            return
        end
    end

    if isempty(ObjectTypename),return,end

    % Calculate mean and standard deviation
    MeasurementMean = zeros(length(Measurements),1);
    MeasurementStd = zeros(length(Measurements),1);

    ImageSet=[];
    for count=FirstImage:LastImage
        ImageSet=[ImageSet; count];
    end
    
    emptymat=0;
    GraphedSamples=[];
    for k = 1:length(ImageSet)
        imagenum=ImageSet(k);
        if ~isempty(Measurements{imagenum})
            GraphedSamples=[GraphedSamples;imagenum];
            MeasurementsMean(k) = mean(Measurements{imagenum}(:,FeatureNo));
            MeasurementsStd(k)  = std(Measurements{imagenum}(:,FeatureNo));
        else
            emptymat=1;
        end
    end
    
    if emptymat
        warnfig=CPwarndlg('There is an empty matrix in your measurement data, so a portion of the measurements will not be taken into account for the graph. This may affect the display of the graph (eg. fewer/no bars). This probably occurred because your custom-chosen data threshold was too stringent. You may consider trying a more lenient threshold.');
        uiwait(warnfig);
    end

    FigHandle=CPfigure;
    
    %%% Do the plotting   
    graph=bar(MeasurementsMean);
    set(graph,'FaceColor',LineColor);
    
    hold on
    for k = 1:length(GraphedSamples)
        plot([k k],[MeasurementsMean(k)-MeasurementsStd(k),MeasurementsMean(k)+MeasurementsStd(k)],'k','linewidth',1)
        plot([k-0.075,k+0.075],[MeasurementsMean(k)-MeasurementsStd(k),MeasurementsMean(k)-MeasurementsStd(k)],'k','linewidth',1)
        plot([k-0.075,k+0.075],[MeasurementsMean(k)+MeasurementsStd(k),MeasurementsMean(k)+MeasurementsStd(k)],'k','linewidth',1)
    end
    hold off

    xlabel(gca,'Image number','Fontname','Helvetica','fontsize',FontSize+2)
    if strcmp(UserAnswers.Logical,'None') ~= 1
        ylabel(gca,{[];str;'for objects where';Thresholdstr;'mean +/- standard deviation'},'fontname','Helvetica','fontsize',FontSize+2)
    else
        ylabel(gca,[str,', mean +/- standard deviation'],'fontname','Helvetica','fontsize',FontSize+2)
    end
    axis([0 length(ImageSet)+1 ylim])
    set(gca,'xtick',[0 1:length(ImageSet) length(ImageSet)+1]);
    set(gca,'xticklabel',[0; ImageSet; LastImage+1]);
    titlestr = str;
    
    %%% Line graph
elseif PlotType == 2

    %%% Extract the measurement
    Measurements = handles.Measurements.(Object).(Feature);
    


    %%% Opens a window that lets the user choose graph settings
    %%% This function returns a UserAnswers structure with the
    %%% information required to carry out the calculations.
    try UserAnswers = UserAnswersWindow(handles);
    catch CPerrordlg(lasterr)
        return
    end

    % If Cancel button pressed, return
    if ~isfield(UserAnswers, 'FirstSample')
        return
    end

    FirstImage=UserAnswers.FirstSample;
    LastImage=UserAnswers.LastSample;

    switch UserAnswers.Color
        case 'Blue'
            LineColor='b';
        case 'Gray'
            LineColor=[.7 .7 .7];
        case 'Green'
            LineColor='g';
        case 'Yellow'
            LineColor='y';
        case 'Magenta'
            LineColor='m';
        case 'Cyan'
            LineColor='c';
        case 'Black'
            LineColor='k';
        case 'White'
            LineColor='w';
        otherwise
            LineColor=[.7 .7 .9];
    end
    
    PlotNum=1;
    %%% Thresholds the data if user chose other than "None"
    %%% Reassigns Measurements
    if strcmp(UserAnswers.Logical,'None') ~= 1
        try 
            msgfig=CPmsgbox('In the following dialog, please select the measurements that the original measurements will be thresholded on.');
            uiwait(msgfig);
            [ObjectTypename,FeatureType,FeatureNum] = CPgetfeature(handles);
            % There's no need to nest the call to CPgetfeature again, because
            % the only errors that can occur in CPgetfeature happen when
            % handles is faulty, but CPgetfeature was called before and handles
            % hasn't been modified. However, an empty check for ObjectTypename
            % is needed. This would happen if the user clicked Cancel.
            [Thresholdstr,Measurements,ObjectTypename,FeatureType,FeatureNum] = Threshold(handles,Measurements,UserAnswers,ObjectTypename,FeatureType,FeatureNum);
        catch CPerrordlg(lasterr)
            return
        end
    end
    
    if isempty(ObjectTypename),return,end
    
    % Calculate mean and standard deviation
    MeasurementMean = zeros(length(Measurements),1);
    MeasurementStd = zeros(length(Measurements),1);
    
    ImageSet=[];
    for count=FirstImage:LastImage
        ImageSet=[ImageSet; count];
    end
   
    emptymat=0;
    for k = 1:length(ImageSet)
        imagenum=ImageSet(k);
        if ~isempty(Measurements{imagenum})
            MeasurementsMean(k) = mean(Measurements{imagenum}(:,FeatureNo));
            MeasurementsStd(k)  = std(Measurements{imagenum}(:,FeatureNo));
        else
            emptymat=1;
        end
    end
    
    if emptymat
        warnfig=CPwarndlg('There is an empty matrix in your measurement data, so a portion of the measurements will not be taken into account for the graph. This may affect the display of the graph (eg. fewer/no lines). This probably occurred because your custom-chosen data threshold was too stringent. You may consider trying a more lenient threshold.');
        uiwait(warnfig);
    end

    %%% Plots a line chart, where the X dimensions are incremented
    %%% from 1 to the number of measurements to be PlotTypeed, and Y is
    %%% the measurement of interest.
    FigHandle=CPfigure;
    hold on
    plot(1:length(MeasurementsMean), MeasurementsMean,'Color',LineColor,'LineWidth',1);

    %%% Plots the Standard deviations as lines, too.
    plot(1:length(MeasurementsMean), MeasurementsMean-MeasurementsStd,':','Color',LineColor);
    plot(1:length(MeasurementsMean), MeasurementsMean+MeasurementsStd,':','Color',LineColor);
    hold off
    axis([0 length(ImageSet)+1 ylim])

    xlabel(gca,'Image number','Fontname','Helvetica','fontsize',FontSize+2)
    if strcmp(UserAnswers.Logical,'None') ~= 1
        ylabel(gca,{[];str;'for objects where';Thresholdstr;'mean +/- standard deviation'},'fontname','Helvetica','fontsize',FontSize+2)
    else
        ylabel(gca,[str,', mean +/- standard deviation'],'fontname','Helvetica','fontsize',FontSize+2)
    end
    
    set(gca,'xtick',[0 1:length(ImageSet) length(ImageSet)+1]);
    set(gca,'xticklabel',[0; ImageSet; LastImage+1]);
    titlestr = str;

    %%% Scatter plot, 1 measurement
elseif PlotType == 3
    
    %%% Extract the measurements
    Measurements = handles.Measurements.(Object).(Feature);

    %%% Opens a window that lets the user choose graph settings
    %%% This function returns a UserAnswers structure with the
    %%% information required to carry out the calculations.
    try UserAnswers = UserAnswersWindow(handles);
    catch CPerrordlg(lasterr)
        return
    end

    % If Cancel button pressed, return
    if ~isfield(UserAnswers, 'FirstSample')
        return
    end

    FirstImage=UserAnswers.FirstSample;
    LastImage=UserAnswers.LastSample;
    
    PlotNum=1;
    %%% Thresholds the data if user chose other than "None"
    %%% Reassigns Measurements
    if strcmp(UserAnswers.Logical,'None') ~= 1
        try 
            msgfig=CPmsgbox('In the following dialog, please select the measurements that the original measurements will be thresholded on.');
            uiwait(msgfig);
            [ObjectTypename,FeatureType,FeatureNum] = CPgetfeature(handles);
            % There's no need to nest the call to CPgetfeature again, because
            % the only errors that can occur in CPgetfeature happen when
            % handles is faulty, but CPgetfeature was called before and handles
            % hasn't been modified. However, an empty check for ObjectTypename
            % is needed. This would happen if the user clicked Cancel.
            [Thresholdstr,Measurements,ObjectTypename,FeatureType,FeatureNum] = Threshold(handles,Measurements,UserAnswers,ObjectTypename,FeatureType,FeatureNum);
        catch CPerrordlg(lasterr)
            return
        end
    end
    
    if isempty(ObjectTypename),return,end

    switch UserAnswers.Color
        case 'Blue'
            LineColor='b';
        case 'Gray'
            LineColor=[.7 .7 .7];
        case 'Green'
            LineColor='g';
        case 'Yellow'
            LineColor='y';
        case 'Magenta'
            LineColor='m';
        case 'Cyan'
            LineColor='c';
        case 'Black'
            LineColor='k';
        case 'White'
            LineColor='w';
        otherwise
            LineColor=[.7 .7 .9];
    end
    
    ImageSet=[];
    for count=FirstImage:LastImage
        ImageSet=[ImageSet;count];
    end
    
    emptymat=0;
    for k = 1:length(ImageSet)
        imagenum=ImageSet(k);
        if isempty(Measurements{imagenum})
            emptymat=1;
        end
    end
    
    if emptymat
        warnfig=CPwarndlg('There is an empty matrix in your measurement data, so a portion of the measurements will not be taken into account for the graph. This may affect the display of the graph (eg. fewer/no data points). This probably occurred because your custom-chosen data threshold was too stringent. You may consider trying a more lenient threshold.');
        uiwait(warnfig);
    end

    FigHandle=CPfigure;
    %%% Plot
    hold on
    for k = FirstImage:LastImage
        if ~isempty(Measurements{k})
            plot(k*ones(length(Measurements{k}(:,FeatureNo))),Measurements{k}(:,FeatureNo),'.k','color',LineColor)
            plot(k,mean(Measurements{k}(:,FeatureNo)),'.r','Markersize',20)
        end
    end
    hold off
    axis([0 length(ImageSet)+1 ylim])

    xlabel(gca,'Image number','Fontname','Helvetica','fontsize',FontSize+2)
    if strcmp(UserAnswers.Logical,'None') ~= 1
        ylabel(gca,{[];str;'for objects where';Thresholdstr},'fontname','Helvetica','fontsize',FontSize+2)
    else
        ylabel(gca,str,'fontname','Helvetica','fontsize',FontSize+2)
    end
    set(gca,'xtick',[0 1:length(ImageSet) length(ImageSet)+1]);
    set(gca,'xticklabel',[0; ImageSet; LastImage+1]);
    titlestr = str;

    %%% Scatter plot, 2 measurements
elseif PlotType == 4

    %%% Extract the measurements
    % Measurements for X axis
    Measurements1 = handles.Measurements.(Object).(Feature);
    % Measurements for Y axis
    Measurements2 = handles.Measurements.(Object2).(Feature2);

    if ModuleFlag == 0
        %%% Calculates some values for the next dialog box.
        TotalNumberImageSets = length(Measurements1);
        TextTotalNumberImageSets = num2str(TotalNumberImageSets);

        %%% Opens a window that lets the user choose graph settings
        %%% This function returns a UserAnswers structure with the
        %%% information required to carry out the calculations.
        try UserAnswers = UserAnswersWindow(handles);
        catch CPerrordlg(lasterr)
            return
        end

        % If Cancel button pressed, return
        if ~isfield(UserAnswers, 'FirstSample')
            return
        end

        FirstImage=UserAnswers.FirstSample;
        LastImage=UserAnswers.LastSample;
        
        PlotNum=1;
        %%% Thresholds the data if user chose other than "None"
        %%% Reassigns Measurements
        if strcmp(UserAnswers.Logical,'None') ~= 1
            try
                msgfig=CPmsgbox('In the following dialog, please select the measurements that the original measurements will be thresholded on.');
                uiwait(msgfig);
                [ObjectTypename,FeatureType,FeatureNum] = CPgetfeature(handles);
                % There's no need to nest the call to CPgetfeature again, because
                % the only errors that can occur in CPgetfeature happen when
                % handles is faulty, but CPgetfeature was called before and handles
                % hasn't been modified. However, an empty check for ObjectTypename
                % is needed. This would happen if the user clicked Cancel.
                [Thresholdstr2,Measurements2,ObjectTypename,FeatureType,FeatureNum] = Threshold(handles,Measurements2,UserAnswers,ObjectTypename,FeatureType,FeatureNum);
                [Thresholdstr1,Measurements1,ObjectTypename,FeatureType,FeatureNum] = Threshold(handles,Measurements1,UserAnswers,ObjectTypename,FeatureType,FeatureNum);
            catch CPerrordlg(lasterr)
                return
            end
        end
        
        if isempty(ObjectTypename),return,end

        switch UserAnswers.Color
            case 'Blue'
                LineColor='b';
            case 'Gray'
                LineColor=[.7 .7 .7];
            case 'Green'
                LineColor='g';
            case 'Yellow'
                LineColor='y';
            case 'Magenta'
                LineColor='m';
            case 'Cyan'
                LineColor='c';
            case 'Black'
                LineColor='k';
            case 'White'
                LineColor='w';
            otherwise
                LineColor=[.7 .7 .9];
        end
        
        FigHandle=CPfigure;
        %%% Plot
        hold on
        emptymat=0;
        for k = FirstImage:LastImage
            if size(Measurements1{k},1) ~= size(Measurements2{k})
                error('The number of objects for the chosen measurements does not match.')
            end
            if ~isempty(Measurements1{k}) & ~isempty(Measurements2{k})
                plot(Measurements1{k}(:,FeatureNo),Measurements2{k}(:,FeatureNo2),'.k', 'color', LineColor)
            else
                emptymat=1;
            end
        end
        hold off
        
    else
        FigHandle=CPfigure;
        hold on
        emptymat=0;
        for k = 1:length(Measurements1)
            if size(Measurements1{k},1) ~= size(Measurements2{k})
                error('The number of objects for the chosen measurements does not match.')
            end
            if ~isempty(Measurements1{k})
                plot(Measurements1{k}(:,FeatureNo),Measurements2{k}(:,FeatureNo2),'.k')
            else
                emptymat=1;
            end
        end
        hold off
    end

    if emptymat
        warnfig=CPwarndlg('There is an empty matrix in your measurement data, so a portion of the measurements will not be taken into account for the graph. This may affect the display of the graph (eg. fewer/no bars). This probably occurred because your custom-chosen data threshold was too stringent. You may consider trying a more lenient threshold.');
        uiwait(warnfig);
    end

    if (length(Feature2) > 10) & strncmp(Feature2,'Intensity_',10)
        str2 = [handles.Measurements.(Object2).([Feature2,'Features']){FeatureNo2},' of ', Object2, ' in ',Feature2(11:end)];
    elseif (length(Feature2) > 8) & strncmp(Feature2,'Texture_',8)
        str2 = [handles.Measurements.(Object2).([Feature2,'Features']){FeatureNo2},' of ', Object2, ' in ',Feature2(11:end)];
    else
        str2 = [handles.Measurements.(Object2).([Feature2,'Features']){FeatureNo2},' of ', Object2];
    end


    if strcmp(UserAnswers.Logical,'None') ~= 1
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
    set(FigHandle,'Numbertitle','off','name',['Plot Measurement: ',get(get(gca,'title'),'string')])
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
set(UserWindow,'units','inches','resize','on','menubar','none','toolbar','none','numbertitle','off','Name','Choose control histogram settings','Color',[.7 .7 .9]);
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
okbutton = uicontrol(UserWindow,'style','pushbutton','String','OK','Fontweight','bold','FontSize',FontSize,'units','inches',...
    'position',[posx 0.1 0.75 0.3],'BackgroundColor',[.7 .7 .9],'Callback','[cobj,cfig] = gcbo;set(cobj,''UserData'',1);uiresume(cfig);clear cobj cfig;','BackgroundColor',[.7 .7 .9]);
cancelbutton = uicontrol(UserWindow,'style','pushbutton','String','Cancel','Fontweight','bold','FontSize',FontSize,'units','inches',...
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
function [Thresholdstr,Measurements,ObjectTypename,FeatureType,FeatureNum] = Threshold(handles,Measurements,UserAnswers,ObjectTypename,FeatureType,FeatureNum);


if isempty(ObjectTypename),return,end
MeasurementToThresholdValueOnName = cat(2,handles.Measurements.(ObjectTypename).([FeatureType,'Features']){FeatureNum},' of ',ObjectTypename);
tmp = handles.Measurements.(ObjectTypename).(FeatureType);
MeasurementToThresholdValueOn = cell(length(tmp),1);
for k = 1:length(tmp)
    MeasurementToThresholdValueOn{k} = tmp{k}(:,FeatureNum);
end

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
    if strcmp(UserAnswers.Logical,'>') == 1 
        newmat=[];
        boolcol=(ListOfMeasurements{CompressedImageNumber,2} > UserAnswers.ThresholdVal);
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
    elseif strcmp(UserAnswers.Logical,'>=') == 1
        newmat=[];
        boolcol=(ListOfMeasurements{CompressedImageNumber,2} >= UserAnswers.ThresholdVal);
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
    elseif strcmp(UserAnswers.Logical,'<') == 1
        newmat=[];
        boolcol=(ListOfMeasurements{CompressedImageNumber,2} < UserAnswers.ThresholdVal);
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
    elseif strcmp(UserAnswers.Logical,'<=') == 1
        newmat=[];
        boolcol=(ListOfMeasurements{CompressedImageNumber,2} <= UserAnswers.ThresholdVal);
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
    else
        newmat=[];
        boolcol=(ListOfMeasurements{CompressedImageNumber,2} == UserAnswers.ThresholdVal);
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
    end

    %%% Increments the CompressedImageNumber.
    CompressedImageNumber = CompressedImageNumber + 1;
end

Measurements=OutputMeasurements';
