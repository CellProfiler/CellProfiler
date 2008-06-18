function TruncatedName = CPtruncatefeaturename(FeatureName,MinStrLen,dlmtr)

if nargin < 3
    dlmtr = '_';
end

if isempty(FeatureName)
    error('FeatureName cannot be an empty string')
end

TruncatedName = '';
FeatureNameSubstrings = textscan(FeatureName,'%s','delimiter',dlmtr);
for idxStr = 1:length(FeatureNameSubstrings{1})
    Str = FeatureNameSubstrings{1}{idxStr};
    TruncatedName = CPjoinstrings(TruncatedName,Str(1:min(length(Str),MinStrLen)));
end

%% Just in case, after all the truncation
if length(TruncatedName) > 63
    TruncatedName = TruncatedName(1:63);
end
