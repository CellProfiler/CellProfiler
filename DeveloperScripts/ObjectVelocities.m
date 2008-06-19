%% Plots location change across frames (quiver plots)

clear
close all

MAX_JUMP = 10;
MIN_TOTAL_DIST = 5;

%% load handles
% load /Volumes/imaging_analysis/2007_07_02Microtube_Motor_Screens_VladimirGelfand/2007_10_11MoviesForAnne/Converted/F05/output_2007_10_26/DefaultOUT__5.mat
% load /Volumes/imaging_analysis/2007_07_02Microtube_Motor_Screens_VladimirGelfand/2007_10_11MoviesForAnne/Converted/F07/output_2007_10_26/DefaultOUT__15.mat
% load /Users/dlogan/Data/2007_10_11MoviesForAnne/Converted/F05/output_2007_11_06/DefaultOUT.mat
% load /Users/dlogan/Data/2007_10_11MoviesForAnne/Converted/F07/output_2007_11_06/DefaultOUT.mat
% load('/Volumes/imaging_analysis-2/2007_07_02Microtube_Motor_Screens_VladimirGelfand/3_11_08_B05/2008_03_28_output/DefaultOUT__1.mat','handles')
load('/Volumes/IMAGING_ANALYSIS/2007_07_02Microtube_Motor_Screens_VladimirGelfand/2008_06_02_wild_type/TIF files wildtype from 31108/D02_31108/2008_06_05_output/DefaultOUT__13.mat')

Objects = handles.Pipeline.TrackObjects.FilteredSpeckles.Labels;
Locations = handles.Pipeline.TrackObjects.FilteredSpeckles.Locations;

%% find max Label across frames
HighestObjNum = max(cellfun(@(x)max(x(:)), Objects));
% HighestObjNum = max(Objects(:));

U = NaN .* ones(HighestObjNum,length(Objects)-1);
V = NaN .* ones(HighestObjNum,length(Objects)-1);

%% Loop Objects, and get difference (velocity) vectors for each frame 2:end

for iObjNum = 1:HighestObjNum
% for iObjNum = 78
    
%     STEP = (length(Objects)-1)-1; %% first and last frames
    STEP = 1; %% every frame
    
    thisFrameLocationAll = NaN .* ones(length(Objects)-1,2);
%     nextFrameLocation = NaN .* ones(length(Objects)-1,2);
    for iFrm = 1:STEP:length(Objects)
        
        thisFrameLocation = Locations{iFrm}(Objects{iFrm} == iObjNum,:);
%         thisFrameLocation = Locations(Objects(iFrm) == iObjNum,:);
        %% If object does not exist in one frame, then skip it
        if isempty(thisFrameLocation)
            break
        else
            %% Keep track of good objects' position
            thisFrameLocationAll(iFrm,:) = thisFrameLocation;
        end
        
        %% If we made it here on the last iteration, then this object exists on all frames
        if iFrm == length(Objects)
            D = diff(thisFrameLocationAll);
            
            %% Regard any large pixel jump in a single frame as an error in TrackObjects labeling -> discard
            if any(abs(D(:))>MAX_JUMP)
                break
            end
            
            %% Require that the total distance travelled must be at least
            %% MIN_TOTAL_DIST
            TotalDist = thisFrameLocationAll(end,:) - thisFrameLocationAll(1,:);
            if norm(TotalDist) < MIN_TOTAL_DIST
                break
            end
            hold on
            h = quiver(thisFrameLocationAll(1:end-1,1),thisFrameLocationAll(1:end-1,2),D(:,1),D(:,2),0);
            set(h,'Linewidth',1)
        end
        
    end
end

axis equal ij
axis([1 512 1 512])
