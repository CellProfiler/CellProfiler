function handles = Relate(handles)

% Help for the Relate module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Assigns relationships: All objects (e.g. speckles) within a parent object
% (e.g. nucleus) become its children.
% *************************************************************************
%
% Allows associating "children" objects with "parent" objects. This is
% useful for counting the number of children associated with each parent,
% and for calculating mean measurement values for all children that are
% associated with each parent. For every measurement that has been made of
% the children objects upstream in the pipeline, this module calculates the
% mean value of that measurement over all children and stores it as a
% measurement for the parent, as "Mean_<child>_<category>_<feature>". 
% For this reason, this module should be placed *after* all Measure modules
% that make measurements of the children objects.
%
% An object will be considered a child even if the edge is the only part
% touching a parent object. If an object is touching two parent objects,
% the objects parent will be the higher numbered parent.
%
% The minimum distances of each child to its parent are also calculated.
% These values are associated with the child objects. If an "Other" object
% is defined (e.g. Nuclei), then distances are calculated to this object
% too, as well as normalized distances.  Normalized distances for each
% child have a range [0 1] and are calculated as:
% (distance to the Parent) / sum(distances to parent and Other object)
%
% To access the Child/Parent label matrix image in downstream modules, use
% the "Other..." method to choose your image and type Parent_Child,
% where 'Parent' and 'Child' are the names of the objects as selected in 
% Relate's first two settings.  For example, if the parent objects are 
% "Cytoplasm" and the child objects are "Speckles", then downstream choose
% "Cytoplasm_Speckles".
%
% Measurement Categories (each with only one Feature):
% Parent, Children, SubObjectFlag, Distance, NormDistance

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What objects are the children objects (subobjects)?
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
SubObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What are the parent objects?
%infotypeVAR02 = objectgroup
%inputtypeVAR02 = popupmenu
ParentName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Do you want to find minimum distances of each child to its parent?
%choiceVAR03 = No
%choiceVAR03 = Yes
%inputtypeVAR03 = popupmenu
FindParentChildDistances = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = (If 'Yes' to above) What other object do you want to find distances to? (Must be one object per parent object, e.g. Nuclei)
%infotypeVAR04 = objectgroup
%choiceVAR04 = Do not use
%inputtypeVAR04 = popupmenu
ParentName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Do you want to generate mean child measurements for all parents?
%choiceVAR05 = No
%choiceVAR05 = Yes
%inputtypeVAR05 = popupmenu
FindMeanMeasurements = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

% Do we want to calculate mean measurements?
wantMeanMeasurements = strncmpi(FindMeanMeasurements,'y',1);

% Do we want to calculate minimum distances?
wantMinDistances = strncmpi(FindParentChildDistances,'y',1);

%%% Retrieves the label matrix image that contains the edited primary
%%% segmented objects.
SubObjectLabelMatrix = CPretrieveimage(handles,['Segmented', SubObjectName],ModuleName,'MustBeGray','DontCheckScale');

%%% Retrieves the label matrix image that contains the edited primary
%%% segmented objects.
ParentObjectLabelMatrix = CPretrieveimage(handles,['Segmented', ParentName{1}],ModuleName,'MustBeGray','DontCheckScale');

%%% Retrieves the label matrix image that contains the edited primary
%%% segmented objects.
if ~strcmp(ParentName{2},'Do not use')

    % Sanity checks
    if strcmp(SubObjectName,ParentName{1}) || strcmp(SubObjectName,ParentName{2})
        CPwarndlg('The Children and at least one of the Parent objects are the same.  Your results may be erroneous.','Relate module')
    end
    if strcmp(ParentName{1},ParentName{2})
        CPwarndlg('The Parent and Other Object are the same.  Your results may be erroneous.','Relate module')
    end

    StepParentObjectLabelMatrix = CPretrieveimage(handles,['Segmented', ParentName{2}],ModuleName,'MustBeGray','DontCheckScale');

    % Sanity check
    if max(ParentObjectLabelMatrix(:)) ~= max(StepParentObjectLabelMatrix(:))
        CPwarndlg(['The number of Parent Objects (' num2str(max(ParentObjectLabelMatrix(:))) ...
            ') does not equal the number of Other objects (' num2str(max(StepParentObjectLabelMatrix(:))) ...
            ') in the Relate Module, Cycle#' num2str(handles.Current.SetBeingAnalyzed) ...
            '.  If the difference is large, this may cause the Relate Module output to be suspect.'],'Relate module')
    end
else
    ParentName = {ParentName{1}};
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

[handles,NumberOfChildren,ParentsOfChildren] = CPrelateobjects(handles,SubObjectName,ParentName{1},...
    SubObjectLabelMatrix,ParentObjectLabelMatrix,ModuleName);
handles = CPaddmeasurements(handles,SubObjectName,'SubObjectFlag',1);

if wantMinDistances
    % Save Distance 'Features'

    % Calcuate the smallest distance from each Child to their Parent
    % If no parent exists, then Distance = NaN

    if isfield(handles.Measurements.(SubObjectName),'Location_Center_X')

        for thisParent = ParentName %% Will need to change if we add more StepParents
            % Calculate perimeters for all parents simultaneously
            DistTransAll = CPlabelperim((CPretrieveimage(handles,['Segmented' thisParent{1}],ModuleName)));
            Dists = zeros(max(SubObjectLabelMatrix(:)), 1);
            if max(ParentsOfChildren) > 0,
                for iParentsOfChildren = 1:max(ParentsOfChildren)
                    % Calculate distance transform to perimeter of Parent objects
                    DistTrans = (bwdist(DistTransAll == iParentsOfChildren));

                    % Get location of each child object
                    ChList = find(ParentsOfChildren == iParentsOfChildren);
                    ChildrenLocationsX = handles.Measurements.(SubObjectName).Location_Center_X{handles.Current.SetBeingAnalyzed}(ChList,:);
                    ChildrenLocationsY = handles.Measurements.(SubObjectName).Location_Center_Y{handles.Current.SetBeingAnalyzed}(ChList,:);
                    roundedChLocX = round(ChildrenLocationsX);
                    roundedChLocY= round(ChildrenLocationsY);
                    idx = sub2ind(size(DistTrans),roundedChLocY(:,1), roundedChLocX(:,1));
                    Dist = DistTrans(idx);
                    Dists(ChList) = Dist;
                end
                handles = CPaddmeasurements(handles,SubObjectName,['Distance_' thisParent{1}], Dists);
            else
                handles = CPaddmeasurements(handles,SubObjectName,['Distance_' thisParent{1}], nan(max(length(NumberOfChildren),1), 1));
            end
        end
    else
        warning('There is no ''Location'' field with which to find subObj to Parent distances')
    end

    % Calculate normalized distances
    % All distances are relative to the *first* parent.
    if length(ParentName) > 1
        FirstParentDist =   handles.Measurements.(SubObjectName).(['Distance_',ParentName{1}]){handles.Current.SetBeingAnalyzed};
        OtherObjDist =      handles.Measurements.(SubObjectName).(['Distance_',ParentName{2}]){handles.Current.SetBeingAnalyzed};
        NormDist = FirstParentDist ./ sum([FirstParentDist OtherObjDist],2);
        NormDist(isnan(NormDist)) = 0;  %% In case sum(Dist,2) == 0 for any reason (no parents/child, or child touching either parent)

        % Save normalized distances
        handles = CPaddmeasurements(handles,SubObjectName, ['NormDistance_',ParentName{1}],NormDist);
    end
end

if wantMeanMeasurements
    % Adds a 'Mean_<SubObjectName>' field to the handles.Measurements structure
    % which finds the mean measurements of all the subObjects that relate to each parent object
    MeasurementFieldnames = fieldnames(handles.Measurements.(SubObjectName))';

    % Some measurments need to be excluded from the per-parent mean calculation
    ExcludedMeasurementsPrefixes = {'SubObjectFlag',...                                             % Child flags, which are a single scalar value (Relate)
        'Parent_','Children_',...                                                                   % Object lists and per-parent counts (CPRelateobjects)
        'Mean_',...                                                                                 % Per-parent mean measurments already calculated (Relate)
        'TrackObjects_Linearity_','TrackObjects_IntegratedDistance_','TrackObjects_Lifetime_'};     % Measurements which are calculated retrospectively (TrackObjects)
    
    if isfield(handles.Measurements.(SubObjectName),['Parent_',ParentName{1}])
        % Why is test line here? Isn't this always the case?  Or is it in case Relate is called twice?- Ray 2007-08-09
        if length(handles.Measurements.(SubObjectName).(CPjoinstrings('Parent_',ParentName{1}))) >= handles.Current.SetBeingAnalyzed
            Parents = handles.Measurements.(SubObjectName).(CPjoinstrings('Parent_',ParentName{1})){handles.Current.SetBeingAnalyzed};
            MeasurementFeatures = fieldnames(handles.Measurements.(SubObjectName));
            for i = 1:length(MeasurementFeatures)
                Fieldname = MeasurementFieldnames{i};
                if any(cell2mat(cellfun(@strncmp,   repmat({Fieldname},[1 length(ExcludedMeasurementsPrefixes)]),...
                                                    ExcludedMeasurementsPrefixes,...
                                                    num2cell(cellfun(@length,ExcludedMeasurementsPrefixes)),'UniformOutput',false)))
                    continue;
                end
                Measurements = handles.Measurements.(SubObjectName).(Fieldname){handles.Current.SetBeingAnalyzed};
                MeanVals = zeros(max(Parents), 1);
                if max(Parents) > 0
                    for j = 1:max(Parents),
                        indices = find(Parents == j);
                        if ~ isempty(indices),
                            MeanVals(j) = mean(Measurements(indices));
                        end
                    end
                end
                handles = CPaddmeasurements(handles, ParentName{1}, CPjoinstrings('Mean',SubObjectName,Fieldname), MeanVals);
            end
        else
            CPwarndlg('The Relate module is attempting to take the mean of a measurement downstream.  Be advised that unless the Relate module is placed *after* all Measurement modules, some ''Mean'' measurements will not be calculated.','Relate Module warning','replace')
        end
    end
end

% Since the label matrix starts at zero, we must include this value in
% the list to produce a label matrix image with children re-labeled to
% their parents values. This does not get saved and is only for display.
if ~isempty(ParentsOfChildren)
    ParentsOfChildrenLM = [0;ParentsOfChildren];
    NewObjectParentLabelMatrix = ParentsOfChildrenLM(SubObjectLabelMatrix+1);
else
    NewObjectParentLabelMatrix = SubObjectLabelMatrix;
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);

ColoredParentLabelMatrixImage = CPlabel2rgb(handles,ParentObjectLabelMatrix);
ColoredSubObjectLabelMatrixImage = CPlabel2rgb(handles,SubObjectLabelMatrix);
ColoredNewObjectParentLabelMatrix = CPlabel2rgb(handles,NewObjectParentLabelMatrix);

if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    fig_h = CPfigure(handles,'Image',ThisModuleFigureNumber);

    
    % Default image
    CPimagesc(ColoredNewObjectParentLabelMatrix,handles,ThisModuleFigureNumber);
    title('New Sub Objects')
    
    % Construct struct which holds images and figure titles
    ud(1).img = ColoredNewObjectParentLabelMatrix;
    ud(2).img = ColoredSubObjectLabelMatrixImage;
    ud(3).img = ColoredParentLabelMatrixImage;

    ud(1).title = ['New Sub Objects, cycle # ',num2str(handles.Current.SetBeingAnalyzed)];
    ud(2).title = ['Original Sub Objects, cycle # ',num2str(handles.Current.SetBeingAnalyzed)];
    ud(3).title = ['Parent Objects, cycle # ',num2str(handles.Current.SetBeingAnalyzed)];

    % Construct uicontrol text, accounting for possible StepParents
    if ~exist('StepParentObjectLabelMatrix','var')
        str = 'New Sub Objects|Original Sub Objects|Parent Objects';
    else
        str = 'New Sub Objects|Original Sub Objects|Parent Objects|StepParent Objects';
        ColoredStepParentObjectLabelMatrix = CPlabel2rgb(handles,StepParentObjectLabelMatrix);
        ud(4).img = ColoredStepParentObjectLabelMatrix;
        ud(4).title = ['StepParent Objects, cycle # ',num2str(handles.Current.SetBeingAnalyzed)];
    end
    
    % Uicontrol for displaying multiple images
    uicontrol(fig_h, 'Style', 'popup',...
        'String', str,...
        'UserData',ud,...
        'units','normalized',...
        'position',[.1 .95 .25 .04],...
        'backgroundcolor',[.7 .7 .9],...
        'Callback', @CP_ImagePopupmenu_Callback);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The label matrix image is saved to the handles structure so it can be
%%% used by subsequent modules.
ColoredNewObjectParentLabelMatrixName = [ParentName{1} '_' SubObjectName];
handles = CPaddimages(handles,ColoredNewObjectParentLabelMatrixName,ColoredNewObjectParentLabelMatrix);