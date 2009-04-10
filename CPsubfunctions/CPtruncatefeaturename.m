function TruncatedName = CPtruncatefeaturename(FeatureName,dlmtr)
% CPtruncatefeaturename 
% Reduce length of delimited text strings to overcome Matlab's 63 character limit
% Finds max length for each substring which will still allow a Matlab aceptable string

% $Revision$

% Starting value for Minimum (Sub)string Length
MinStrLenInit = namelengthmax;

if nargin < 2
    dlmtr = '_';
end

if isempty(FeatureName)
    error('FeatureName cannot be an empty string')
end

% Loop through Minimum String Length, from large to small, and stop when 
% length < 64
for MinStrLen = MinStrLenInit:-1:1
    TruncatedName = '';
    FeatureNameSubstrings = textscan(FeatureName,'%s','delimiter',dlmtr);
    for idxStr = 1:length(FeatureNameSubstrings{1})
        Str = FeatureNameSubstrings{1}{idxStr};
        TruncatedName = CPjoinstrings(TruncatedName,Str(1:min(length(Str),MinStrLen)));
    end
  
    if length(TruncatedName) <= namelengthmax
        if length(TruncatedName) ~= length(FeatureName)
            msgboxtitle = 'Truncation of feature name';
            h = findobj(allchild(0),'name',msgboxtitle);
            if isempty(h)
                h = CPwarndlg({['The following feature names have exceeded Matlab''s ' num2str(namelengthmax) ' character limit.'],...
                                'The original feature name is shown, followed by the truncated name.',...
                                'It is possible this will confuse post-hoc analyses, so you might avoid this by shortening this feature name within Cellprofiler.',...
                                ' ',...
                                [FeatureName,': ',TruncatedName]},msgboxtitle);
                
                set(h,'visible','off');
            else
                hdl_text = get(findobj(h,'type','text','-depth',inf),'string');
                hdl_text{end+1} = [FeatureName,': ',TruncatedName];
                delete(h);
                h = CPwarndlg(hdl_text,msgboxtitle);
                set(h,'visible','off');
            end
            % Indent the last line with the truncated name
            hdl_text = get(findobj(h,'type','text','-depth',inf),'string');
            hdl_text{end} = ['  ',TruncatedName];
            set(findobj(h,'type','text','-depth',inf),'string',hdl_text);
            set(h,'visible','on');
        end
        return
    end
end