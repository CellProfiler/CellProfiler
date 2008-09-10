function CP_OrigNewImage_Callback(hObject,eventdata)

%% Callback used to show "Before" and "After" figures.  
%% An on/off button controls the display
%%
%% Originally used in OverlayOutlines.m.
%% 
%% An example uicontrol object to put in the calling function, 
%% after the CPfigure and CPimagesc functions is below.  It will 
%% place  the on/off button on the left side of the axis:
%%
%%     uicontrol(FigHandle,'units','normalized','position',[.01 .5 .06 .04],'string','off',...
%%        'UserData',{OrigImage NewImage},'backgroundcolor',[.7 .7 .9],...
%%        'Callback',@CP_OrigNewImage_Callback);

% $Revision$

string = lower(get(hObject,'string'));
UserData = get(hObject,'UserData'); 
h_image = findobj(gcbf,'type','image');

switch string,
    case 'off',
        set(h_image,'cdata',UserData{1});
        set(hObject,'string','on');
    case 'on',
        set(h_image,'cdata',UserData{2});
        set(hObject,'string','off');
    otherwise
        set(hObject,'string','on');
end
    
