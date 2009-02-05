function str = CPreplacemetadata(handles,str)

%% Substitute Metadata tokens if found
token = regexp(str, '\(\?[<](?<token>.+?)[>]\)','tokens');
token = [token{:}];
assert(numel(token) == 0 || numel(token) == 1, ...
    ['The number of regular expression tokens found was not 0 or 1.  '...
    'Please adjust your token settings, i.e. (?<token>)'])
if isfield(handles.Measurements.Image,['Metadata_' token{1}]);
    replace_string = handles.Measurements.Image.(['Metadata_' token{1}]);
    str = regexprep(str, '\(\?[<](?<token>.+?)[>]\)', replace_string);
end