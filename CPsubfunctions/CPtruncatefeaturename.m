function TruncatedName = CPtruncatefeaturename(FeatureName,dlmtr)
% CPtruncatefeaturename 
% Reduce length of delimited text strings to overcome Matlab's 63 character limit
%% Finds max length for each substring which will still allow a Matlab aceptable string

% $Revision$

%% Starting value for Minimum (Sub)string Length
MinStrLenInit = 20;

if nargin < 2
    dlmtr = '_';
end

if isempty(FeatureName)
    error('FeatureName cannot be an empty string')
end

%% Loop through Minimum String Length, from large to small, and stop when 
%% length < 64
for MinStrLen = MinStrLenInit:-1:1
    TruncatedName = '';
    FeatureNameSubstrings = textscan(FeatureName,'%s','delimiter',dlmtr);
    for idxStr = 1:length(FeatureNameSubstrings{1})
        Str = FeatureNameSubstrings{1}{idxStr};
        TruncatedName = CPjoinstrings(TruncatedName,Str(1:min(length(Str),MinStrLen)));
    end
    
    if length(TruncatedName) < 64
        return
    end
end

