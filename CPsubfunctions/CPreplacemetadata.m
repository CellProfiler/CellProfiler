function [str,doTokensExist] = CPreplacemetadata(handles,str)

% Substitute Metadata tokens if found
token = regexp(str, '\(\?[<](?<token>.+?)[>]\)','tokens');
token = [token{:}];
doTokensExist = ~isempty(token);

for i = 1:numel(token)
    if isfield(handles.Measurements.Image,['Metadata_' token{i}]);
        replace_string = handles.Measurements.Image.(['Metadata_' token{i}]);
        str = regexprep(str, ['\(\?[<](' token{i} ')[>]\)'], replace_string);
    end
end