function FeatureName = CPgetfeaturenamesfromnumbers(handles,ObjectName,Category,FeatureNumber,Image,TextureScale)
%% Get FeatureNames from FeatureNumbers
%%
%% handles = CPgetfeaturenamesfromnumbers(handles, ObjectName, Category, FeatureNumber)
%% handles = CPgetfeaturenamesfromnumbers(handles, ObjectName, Category, FeatureNumber, Image)
%% handles = CPgetfeaturenamesfromnumbers(handles, ObjectName, Category, FeatureNumber, Image, TextureScale)
%%
%%  where   handles is the handles structure
%%          ObjectNamem & Category are character strings
%%          FeatureNumber, TextureScale are positive integers
%%          Image can be can be a string (Image) or Numeric (Texture)
%%
%% where we are finding FeatureName from
%%  handles.Measurements.ObjectName.Category_FeatureName (for Non-image/non-texture) OR
%%  handles.Measurements.ObjectName.Category_FeatureName_Image (for Image) OR
%%  handles.Measurements.ObjectName.Category_FeatureName_Image_TextureScale (for Texture)

%% Note!!  We are relying on the fact that the features should have been 
%% added in the order matching that of each Module's Help text!

error(nargchk(4, 6, nargin, 'string'))

%% Parse inputs
if isempty(Image), Image = ''; end
if isempty(TextureScale), TextureScale = ''; end
if strcmp(Category,'Intensity')
    TextureScale = '';
elseif ~strcmp(Category,'Texture')
    TextureScale = ''; 
    Image = '';
end

AllFieldNames = fieldnames(handles.Measurements.(ObjectName));
CurrentMeasure = [CPjoinstrings(Category,'.*',Image,num2str(TextureScale)),'$'];
FieldnumsCategoryCell = regexp(AllFieldNames,CurrentMeasure);
FieldnumsCategory = find(~cellfun('isempty',FieldnumsCategoryCell));
FeatureName = AllFieldNames(FieldnumsCategory(FeatureNumber));

%% CHECK: There should be one Measurement that fulfills the above criteria
if length(FeatureName) < 1
    error('No Measurements found.  Please check that the measurement exists in the pipeline above this module.')
elseif length(FeatureName) > 1
    error('Too many Measurements fit your criteria.  Please check your settings.')
end

FeatureName = char(FeatureName);