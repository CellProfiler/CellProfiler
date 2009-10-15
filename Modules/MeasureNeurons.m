function handles = MeasureNeurons(handles,varargin)

% Help for the MeasureNeurons module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% This module will measure branching info of skelton objects from seed points.
%
%

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%

drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the seed objects (e.g. soma)?
%infotypeVAR01 = objectgroup
SeedObjects = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the skeleton image? (This is usually a dendrite or axon image saved from the Morph module)
%infotypeVAR02 = imagegroup
SkeletonName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%%%VariableRevisionNumber = 1


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the incoming skeleton image and assigns it to a variable.
SkeletonImg = CPretrieveimage(handles,SkeletonName,ModuleName,'MustBeBinary','CheckScale');

%%% Reads (opens) the incoming image and assigns it to a variable.
SeedObjectsLabelMatrix = CPretrieveimage(handles,['Segmented', SeedObjects],ModuleName,'DontCheckColor','DontCheckScale',size(SkeletonImg));


%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

% [branch_points, branch_dist_tab] = CPbranch_dist(Skeleton,SeedObjectsLabelMatrix);

% function [branch_points, branch_dist_tab] = CPbranch_dist(SkeletonImg,SeedObjectsLabelMatrix)
% Finds skeleton branch point distances from SeedObjectsLabelMatrix

%assume SkeletonImg and SeedObjectsLabelMatrix labeled the same

%% create new skeleton with 'holes' at nuclei
combined_skel=or(SkeletonImg,SeedObjectsLabelMatrix);
seed_center=imerode(SeedObjectsLabelMatrix,strel('disk',2));
combined_skel=xor(combined_skel,seed_center);
seed_edge=xor(SeedObjectsLabelMatrix,seed_center);

%% thin a second time to avoid faulty branch points
combined_skel=bwmorph(combined_skel,'thin',Inf);

[IgnoreLabels, DistanceMap] = IdentifySecPropagateSubfunction(SeedObjectsLabelMatrix, zeros(size(SeedObjectsLabelMatrix)), max(combined_skel,SeedObjectsLabelMatrix)> 0, 1.0);

%% Change background from white -> black
DistanceMap(isinf(DistanceMap)) = -Inf; 
branch_points = bwmorph(combined_skel,'branchpoints');

%% Remove branch points that fall on objects removed by
% IdentifySecPropagateSubfunction and thus appear orphaned
branch_points = and(IgnoreLabels,branch_points);

%% transfer labels to branch point image
num_seeds=max(SeedObjectsLabelMatrix(:));
% lab_branch_map=max(SkeletonImg,SeedObjectsLabelMatrix).*branch_points; %% original code
lab_branch_map=max(IgnoreLabels,SeedObjectsLabelMatrix).*branch_points;
for i=num_seeds:-1:1
    branch_dist_tab{i}=DistanceMap(lab_branch_map==i);
    NumTrunks(i) = length(branch_dist_tab{i} == 0);
    NumNonTrunkBranches(i) = length(branch_dist_tab{i} > 0);
end


%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%

trunks = branch_points & (DistanceMap == 0);
nonZeroBranches = branch_points & (DistanceMap > 0);

% Skeleton is blue
r = zeros(size(combined_skel));
g = zeros(size(combined_skel));
b = combined_skel;

%% "Trunk" branchpoints (Distance = 0 branchpoints, i.e. those that fall on the seed object) 
%% are Red
r(trunks) = 1;
g(trunks) = 0;
b(trunks) = 0;

%% NonTrunk branchpoints (Distance > 0 branchpoints, i.e. those that fall beyond the seed object) 
%% are Green
r(nonZeroBranches) = 0;
g(nonZeroBranches) = 1;
b(nonZeroBranches) = 0;

visRGB = cat(3,r,g,b);

% r_nonzero_branches(branch_points) = 0;
% g_nonzero_branches(branch_points & (DistanceMap > 0)) = 1;
% b_nonzero_branches(branch_points) = 0;
% 
% visRGB = cat(3,r_combined_skel,g_combined_skel,b_combined_skel);
% visRGB = cat(3,visRGB(:,:,1)+r_nonzero_branches, visRGB(:,:,2)+g_nonzero_branches, visRGB(:,:,3)+b_nonzero_branches);

drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this module.
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
%         CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
    end
    %%% A subplot of the figure window is set to display the original image.
%     hAx=subplot(2,1,1,'Parent',ThisModuleFigureNumber); 
    [hImage,hAx] = CPimagesc(visRGB,handles,ThisModuleFigureNumber);
    title(hAx,['Branchpoints (Distance=0 are red, other branches green) cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves measurements to the handles structure.
%   handles = CPaddmeasurements(handles,ObjectName,FeatureName,Data);
handles = CPaddmeasurements(handles, SkeletonName, ...
    'Neurons_NumTrunks', NumTrunks');
handles = CPaddmeasurements(handles, SkeletonName, ...
    'Neurons_NumNonTrunkBranches', NumNonTrunkBranches');

        
% %% Save Branchpoint Image
BranchpointLabelSeedSkeletonName = [SeedObjects '_' SkeletonName];
handles = CPaddimages(handles,BranchpointLabelSeedSkeletonName,visRGB);