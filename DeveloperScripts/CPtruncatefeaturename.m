TruncatedName = CPtruncatefeaturename(FeatureName,StrLen,dlmtr);

if ~exist(dlmtr,'var')
    dlmtr = '_';
end

FeatureNameSubstrings = textscan(FeatureName,'%s','delimiter',dlmtr);
for idxStr = 1:length(FeatureNameSubstrings{1})
    Str = FeatureNameSubstrings{1}{idxStr};
    TruncatedName{idxStr} = Str(1:min(length(Str),MinStrLen));
end