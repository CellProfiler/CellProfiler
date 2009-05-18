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
            
            hdl_dlg = findobj(allchild(0),'name',msgboxtitle);
            if isempty(hdl_dlg)
                warningtextstr = {['The following feature names have exceeded Matlab''s ' num2str(namelengthmax) ' character limit.'],...
                                'The original feature name is shown, followed by the truncated name.',...
                                'It is possible this will confuse post-hoc analyses, so you might avoid this by shortening this feature name within Cellprofiler.',...
                                ' ',...
                                [FeatureName,': ',TruncatedName]};
                hdl_dlg = CPwarndlg(warningtextstr,msgboxtitle,'replace');
                hdl_text = findobj(hdl_dlg,'type','text','-depth',inf);
                set(hdl_text,'visible','off','units','normalized'); 
                p = get(hdl_text,'extent');
                uicontrol('parent',hdl_dlg,'style','edit','string',warningtextstr,'tag',msgboxtitle,...
                    'units','normalized','position',[p(1:2) 1-p(1) p(4)],'enable','inactive','max',1.001,'min',0);
            else
                hdl_uitext = findobj(hdl_dlg,'tag',msgboxtitle);
                warningtextstr = get(hdl_uitext,'string');
                warningtextstr{end+1} = [FeatureName,': ',TruncatedName];
                set(hdl_uitext,'string',warningtextstr);
            end
        end
        return
    end
end