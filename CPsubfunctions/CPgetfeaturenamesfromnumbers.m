function FeatureName = CPgetfeaturenamesfromnumbers(handles,ObjectName,Category,FeatureNumberOrName,Image,TextureScale)
% Get FeatureNames from FeatureNumbers or validate if FeatureName
%
% handles = CPgetfeaturenamesfromnumbers(handles, ObjectName, Category, FeatureNumberOrName)
% handles = CPgetfeaturenamesfromnumbers(handles, ObjectName, Category, FeatureNumberOrName, Image)
% handles = CPgetfeaturenamesfromnumbers(handles, ObjectName, Category, FeatureNumberOrName, Image, TextureScale)
%
%  where   handles is the handles structure
%          ObjectName & Category are character strings
%          FeatureNumberOrName is a positive integer or a feature name
%          TextureScale is a positive integers
%          Image can be can be a string (Image) or Numeric (Texture)
%
% where we are finding FeatureName from
%  handles.Measurements.ObjectName.Category_FeatureName (for Non-image/non-texture) OR
%  handles.Measurements.ObjectName.Category_FeatureName_Image (for Image) OR
%  handles.Measurements.ObjectName.Category_FeatureName_Image_TextureScale (for Texture)

% Note!!  We are relying on the fact that the features should have been 
% added in the order matching that of each Module's Help text!

error(nargchk(4, 6, nargin, 'string'))

% Parse inputs
if ~exist('Image','var'), Image = ''; end
if ~exist('TextureScale','var'), TextureScale = ''; end
if strcmp(Category,'Intensity')
    TextureScale = '';
elseif ~strcmp(Category,'Texture')
    TextureScale = ''; 
    Image = '';
end

AllFieldNames = fieldnames(handles.Measurements.(ObjectName));
if isnumeric(FeatureNumberOrName)
    CurrentMeasure = [CPjoinstrings(Category,'.*',Image,num2str(TextureScale)),'$'];
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
    	if ~ isempty(TextureScale)
            CurrentMeasure = CPjoinstrings(CurrentMeasure,'_',num2str(TextureScale));
    	end
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