function [str,doTokensExist] = CPreplacemetadata(handles,str)

% Substitute Metadata tokens if found
token = regexp(str, '\(\?[<](?<token>.+?)[>]\)','tokens');
token = [token{:}];
doTokensExist = ~isempty(token);

for i = 1:numel(token)
    if isfield(handles.Pipeline.CurrentMetadata,token{i})
        if ~isempty(handles.Pipeline.CurrentMetadata.(token{i}))
            % Grab the token field value from handles.Pipeline metadata
            replace_string = handles.Pipeline.CurrentMetadata.(token{i});
        else
            % If we end up here, a token wasn't found even though there
            % should be one. This indicates an empty filename (i.e., no
            % image) and if we're using image groups, we might be able to
            % figure out the proper token from that
            if isfield(handles.Pipeline,'ImageGroupFields') && any(ismember(handles.Pipeline.ImageGroupFields,token{i}))
                idx = find(ismember(handles.Pipeline.ImageGroupFields,token{i}));
                replace_string = handles.Pipeline.GroupFileList{handles.Pipeline.CurrentImageGroupID}.Fields{idx};
            else
                replace_string = '';    % Give up and leave it blank
            end
        end
        str = regexprep(str, ['\(\?[<](' token{i} ')[>]\)'], replace_string);
    end
end