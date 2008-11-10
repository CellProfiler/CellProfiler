function FeatureName = CPgetfeaturenamesfromnumbers(handles,ObjectName,Category,FeatureNumberOrName,Image,SizeScale)
% Get FeatureNames from FeatureNumbers or validate if FeatureName
%
% handles = CPgetfeaturenamesfromnumbers(handles, ObjectName, Category, FeatureNumberOrName)
% handles = CPgetfeaturenamesfromnumbers(handles, ObjectName, Category, FeatureNumberOrName, Image)
% handles = CPgetfeaturenamesfromnumbers(handles, ObjectName, Category, FeatureNumberOrName, Image, SizeScale)
%
%  where   handles is the handles structure
%          ObjectName & Category are character strings
%          FeatureNumberOrName is a positive integer or a feature name
%          SizeScale is a positive integer
%          Image can be can be a string (Image) or Numeric (Texture)
%
% where we are finding FeatureName from
%  handles.Measurements.ObjectName.Category_FeatureName (for Non-image/non-texture) OR
%  handles.Measurements.ObjectName.Category_FeatureName_Image (for Image) OR
%  handles.Measurements.ObjectName.Category_FeatureName_Image_SizeScale (for Texture)

% $Revision$

% Note!!  We are relying on the fact that the features should have been 
% added in the order matching that of each Module's Help text!

error(nargchk(4, 6, nargin, 'string'))
%% If string is really a number, then makes its class numeric
if ~ isnan(str2double(FeatureNumberOrName))
    FeatureNumberOrName = str2double(FeatureNumberOrName);
        if strcmp(Category,'Count')
            error('The Category ''Count'' requires that you type in the object name in the setting entry for ''What feature do you want to use?'' ')
        end
end
% Parse inputs
if ~exist('Image','var'), Image = ''; end
if ~exist('SizeScale','var'), SizeScale = ''; end
if strcmp(Category,'AreaShape') || ...
        strcmp(Category,'Ratio') || ...
        strcmp(Category,'Math') || ...
        strcmp(Category,'Parent') || ...
        strcmp(Category,'Count') 
    SizeScale = '';
    Image = '';
elseif strcmp(Category,'Intensity') || ...
        strcmp(Category,'ImageQuality') || ...
        strcmp(Category,'Granularity') || ...
        strcmp(Category,'AreaOccupied')
    SizeScale = '';
elseif strcmp(Category,'Texture') ...
        || strcmp(Category,'RadialDistribution')
%% Nothing to do.  These should have all arguments specified
elseif strcmp(Category,'Neighbors')
    Image = '';
elseif strcmp(Category,'Correlation')
%% Children is special because 'Count' is added in CPrelateobjects
elseif strcmp(Category,'Children') 
    Image = 'Count';
    SizeScale = '';
else 
    %% TODO: SaturationBlur
    error('Measurement category could not be found.')
end

AllFieldNames = fieldnames(handles.Measurements.(ObjectName));
if isnumeric(FeatureNumberOrName)
    CurrentMeasure = [CPjoinstrings(Category,'.*',Image,num2str(SizeScale)),'$'];
    FieldnumsCategoryCell = regexp(AllFieldNames,CurrentMeasure);
    FieldnumsCategory = find(~cellfun('isempty',FieldnumsCategoryCell));
    % Could do error checking here, since the next line is where this subfn usually errors
    % (if it can't find a FeatureName), but we ought to use 'try/catch' in the 
    % calling function, for better error handling
    FeatureName = AllFieldNames(FieldnumsCategory(FeatureNumberOrName));
else
    CurrentMeasure = CPjoinstrings(Category,'_',FeatureNumberOrName);
    if ~ isempty(Image)
    	CurrentMeasure = CPjoinstrings(CurrentMeasure,'_',Image);
    end
    if ~ isempty(SizeScale)
        CurrentMeasure = CPjoinstrings(CurrentMeasure,'_',num2str(SizeScale));
    end
    Matches=strcmp(AllFieldNames,CurrentMeasure);
    FeatureNumber=find(Matches);
    if length(FeatureNumber) < 1
        error('No measurement with name "%s" found.', CurrentMeasure);
    end
    FeatureName=AllFieldNames(FeatureNumber);
end

% CHECK: There should be one Measurement that fulfills the above criteria
if length(FeatureName) < 1
    error('No Measurements found.  Please check that the measurement exists in the pipeline above this module.')
elseif length(FeatureName) > 1
    error('Too many Measurements fit your criteria.  Please check your settings.')
end

FeatureName = char(FeatureName);
