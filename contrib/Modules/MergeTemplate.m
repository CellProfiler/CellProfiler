function handles = MergeTemplate(handles)

% Help for the Match Template module:
% Category: Object Processing
% this time we ignore individual cells

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What templates do you want to merge (highest in rank)?
%choiceVAR01 = do not use
%infotypeVAR01 = objectgroup
ObjectName1 = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What templates do you want to merge (intermediate in rank)?
%choiceVAR02 = do not use
%infotypeVAR02 = objectgroup
ObjectName2 = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What templates do you want to merge (lowest in rank)?
%choiceVAR03 = do not use
%infotypeVAR03 = objectgroup
ObjectName3 = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = How do you want to call the merged objects?
%defaultVAR04 = Ring
%infotypeVAR04 = objectgroup indep
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What do you want to call the outlines of the matched templates (optional)?
%defaultVAR05 = Do not save
%infotypeVAR05 = outlinegroup indep
SaveOutlines = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%%%VariableRevisionNumber = 1

if strcmp(ObjectName1, 'do not use')
    error (['Image processing was canceled in the ', ModuleName, 'module. Not enough templates to merge (2-3)'])
end

if strcmp(ObjectName2, 'do not use')
    error (['Image processing was canceled in the ', ModuleName, 'module. Not enough templates to merge (2-3)'])
end

z = handles.Current.SetBeingAnalyzed;

%%% Set up the window for displaying the results
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber);
    CPfigure(handles,'Text',ThisModuleFigureNumber);
    columns = 1;
end
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Retrieve LabelMatrixImages               %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Retrieve the label matrix image that contains the segmented objects of previous modules.

LabelMatrixImage1 = CPretrieveimage(handles,['Segmented', ObjectName1],ModuleName,'MustBeGray','DontCheckScale');
[YLength, XLength] = size (LabelMatrixImage1);

LabelMatrixImage2 = CPretrieveimage(handles,['Segmented', ObjectName2],ModuleName,'MustBeGray','DontCheckScale');
if any(size(LabelMatrixImage2) ~= size(LabelMatrixImage1))
    error([' The size of the image you want to merge is not the same as the size of the image from which the ',ObjectName1,' templates were identified.'])
end

if ~strcmp(ObjectName3, 'do not use')
    LabelMatrixImage3 = CPretrieveimage(handles,['Segmented', ObjectName3],ModuleName,'MustBeGray','DontCheckScale');
    if any(size(LabelMatrixImage3) ~= size(LabelMatrixImage1))
        error([' The size of the image you want to merge is not the same as the size of the image from which the ',ObjectName3,' templates were identified.'])
    end
end

Picture1=LabelMatrixImage1;
Picture2=LabelMatrixImage2;
if ~strcmp(ObjectName3, 'do not use')
    Picture3=LabelMatrixImage3;
else
    Picture3=zeros(YLength, XLength);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CALCULATE IMAGES                              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% we extract the names of the features. we assume, that all MatchTemplates
% did work with the same basic features
FieldNamesList  = fieldnames (handles.Measurements.(ObjectName1));
FieldNamesList  = FieldNamesList(~cellfun('isempty', regexp(FieldNamesList, '^MatchTempl_')));
FieldNamesList  = FieldNamesList(~cellfun('isempty', regexp(FieldNamesList, 'Features$')));
BasicFeaturesName1 = FieldNamesList{1};
FeaturesName1   = BasicFeaturesName1(1:length(BasicFeaturesName1)-8);

% the same for object two
FieldNamesList  = fieldnames (handles.Measurements.(ObjectName2));
FieldNamesList  = FieldNamesList(~cellfun('isempty', regexp(FieldNamesList, '^MatchTempl_')));
FieldNamesList  = FieldNamesList(~cellfun('isempty', regexp(FieldNamesList, 'Features$')));
BasicFeaturesName2 = FieldNamesList{1};
FeaturesName2   = BasicFeaturesName2(1:length(BasicFeaturesName1)-8);

% if present the same for object three
if ~strcmp(ObjectName3, 'do not use')
    FieldNamesList  = fieldnames (handles.Measurements.(ObjectName3));
    FieldNamesList  = FieldNamesList(~cellfun('isempty', regexp(FieldNamesList, '^MatchTempl_')));
    FieldNamesList  = FieldNamesList(~cellfun('isempty', regexp(FieldNamesList, 'Features$')));
    BasicFeaturesName3 = FieldNamesList{1};
    FeaturesName3   = BasicFeaturesName3(1:length(BasicFeaturesName1)-8);
end

% in order not to loose the features of the matched templates we generate images of all features
BasicFeatures = handles.Measurements.(ObjectName1).(BasicFeaturesName1);
for x1 = 1:length(BasicFeatures)
    FeatureMatrixImage1{x1} = zeros(YLength, XLength);
    for x2=1 : max(max(LabelMatrixImage1))
        FeatureMatrixImage1{x1}(LabelMatrixImage1==x2) = handles.Measurements.(ObjectName1).(FeaturesName1){z}(x2, x1);
    end
end
for x1 = 1:length(BasicFeatures)
    FeatureMatrixImage2{x1} = zeros(YLength, XLength);
    for x2=1 : max(max(LabelMatrixImage2))
        FeatureMatrixImage2{x1}(LabelMatrixImage2==x2) = handles.Measurements.(ObjectName2).(FeaturesName2){z}(x2, x1);
    end
end
if ~strcmp(ObjectName3, 'do not use')
    for x1 = 1:length(BasicFeatures)
        FeatureMatrixImage3{x1} = zeros(YLength, XLength);
        for x2=1 : max(max(LabelMatrixImage3))
            FeatureMatrixImage3{x1}(LabelMatrixImage3==x2) = handles.Measurements.(ObjectName3).(FeaturesName3){z}(x2, x1);
        end
    end
end 
drawnow

% now we test for overlap
% we first transform the LabelMatrixImage into a binary image
BinMatrixImage1 = LabelMatrixImage1 > 0;
BinMatrixImage2 = LabelMatrixImage2 > 0;
TestLabelMatrixImage2 = LabelMatrixImage2;
% we delete all non overlapping areas, the resulting image only contains
% the respective number if the images had overlapped
TestLabelMatrixImage2(not(BinMatrixImage1)) = 0; 
for x1 = 1:max(max(LabelMatrixImage2))
    if max(max(TestLabelMatrixImage2==x1))
        BinMatrixImage2(LabelMatrixImage2==x1)=0;
    end
end

LabelMatrixImage2(not(BinMatrixImage2)) = 0;
LabelMatrixImage1 = LabelMatrixImage1 + LabelMatrixImage2;
for x1 = 1:length(BasicFeatures)
    FeatureMatrixImage2{x1}(not(BinMatrixImage2)) = 0;
    FeatureMatrixImage1{x1} = FeatureMatrixImage1{x1} + FeatureMatrixImage2{x1};
end

% if the object is present, we do the same for objects 1 and 3
if ~strcmp(ObjectName3, 'do not use')
    BinMatrixImage1 = LabelMatrixImage1 > 0;
    BinMatrixImage3 = LabelMatrixImage3 > 0;
    TestLabelMatrixImage3 = LabelMatrixImage3;
    % we delete all non overlapping areas, the resulting image only contains
    % the respective number if the images had overlapped
    TestLabelMatrixImage3(not(BinMatrixImage1)) = 0; 
    for x1 = 1:max(max(LabelMatrixImage3))
        if max(max(TestLabelMatrixImage3==x1))
            BinMatrixImage3(LabelMatrixImage3==x1)=0;
        end
    end
    LabelMatrixImage3(not(BinMatrixImage3)) = 0;
    LabelMatrixImage1 = LabelMatrixImage1 + LabelMatrixImage3;
    for x1 = 1:length(BasicFeatures)
        FeatureMatrixImage3{x1}(not(BinMatrixImage3)) = 0;
        FeatureMatrixImage1{x1} = FeatureMatrixImage1{x1} + FeatureMatrixImage3{x1};
    end
end

[FinalSegmentedImage, ObjectCount] = bwlabel (LabelMatrixImage1);

%%% Display
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber);
    drawnow
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    subplot (2,2,1);
    imagesc(Picture1); title ('Object1')
    subplot (2,2,2);
    imagesc(Picture2); colormap ('jet'), title ('Object2')
    subplot (2,2,3);
    imagesc(Picture3); title ('Object3')
    subplot (2,2,4);
    imagesc(FinalSegmentedImage); title('MergedImage')
end

%%% Saves images to the handles structure so they can be saved to the hard
%%% drive, if the user requested.
if ~strcmpi(SaveOutlines,'Do not save')
    FinalOutline = bwperim (FinalSegmentedImage);
    try    handles.Pipeline.(SaveOutlines) = FinalOutline;
    catch error(['The object outlines were not calculated by the ', ModuleName, ' module, so these images were not saved to the handles structure. The Save Images module will therefore not function on these images. This is just for your information - image processing is still in progress, but the Save Images module will fail if you attempted to save these images.'])
    end
end
drawnow         

% for some later modules of CP we need the following images, even though
% they do not make sense in the context of the MatchTemplate module
% FinalLabelMatrixImage = FinalSegmentedImage;
UneditedLabelMatrixImage = FinalSegmentedImage;
SmallRemovedLabelMatrixImage = FinalSegmentedImage;

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['Segmented', ObjectName];
handles.Pipeline.(fieldname) = FinalSegmentedImage;

%%% Saves the segmented image, not edited for objects along the edges or
%%% for size, to the handles structure.
% basically only necessary for following modules (enlarge...)
fieldname = ['UneditedSegmented',ObjectName];
handles.Pipeline.(fieldname) = UneditedLabelMatrixImage;

%%% Saves the segmented image, only edited for small objects, to the
%%% handles structure.
% basically only necessary for following modules (enlarge...)
fieldname = ['SmallRemovedSegmented',ObjectName];
handles.Pipeline.(fieldname) = SmallRemovedLabelMatrixImage;

%%% Saves the ObjectCount, i.e., the number of matched objects.
if ~isfield(handles.Measurements.Image,'ObjectCountFeatures')
    handles.Measurements.Image.ObjectCountFeatures = {};
    handles.Measurements.Image.ObjectCount = {};
end
column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ObjectCountFeatures,ObjectName)));
if isempty(column)
    handles.Measurements.Image.ObjectCountFeatures(end+1) = {ObjectName};
    column = length(handles.Measurements.Image.ObjectCountFeatures);
end
handles.Measurements.Image.ObjectCount{handles.Current.SetBeingAnalyzed}(1,column) = ObjectCount;
%%% Saves the location of each segmented object
handles.Measurements.(ObjectName).LocationFeatures = {'CenterX','CenterY'};
tmp = regionprops(FinalSegmentedImage,'Centroid');
Centroid = cat(1,tmp.Centroid);
       if isempty(Centroid)
            Centroid = [0 0];
       end
handles.Measurements.(ObjectName).Location(handles.Current.SetBeingAnalyzed) = {Centroid};

%%% Save measurements
for f=1:length(BasicFeatures)
    Features(1,f) = 0;
end
for x1=1 : ObjectCount
    for x2=1:length(BasicFeatures)
        Features(x1, x2) = max(max(FeatureMatrixImage1{x2}(FinalSegmentedImage==x1)));
    end
end
handles.Measurements.(ObjectName).('MatchTempl_MergedImageFeatures') = BasicFeatures;
handles.Measurements.(ObjectName).('MatchTempl_MergedImage')(handles.Current.SetBeingAnalyzed) = {Features};

return