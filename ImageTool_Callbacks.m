function ImageTool_Callbacks(action)
%
% This function contains the Callback functions for the Image Tools
%

% Get the handle to the Image Tool window. The handle to the image
% on which the functions should operate is stored in the UserData property
% of the Image Tool window.
[foo, ITh] = gcbo;

if ishandle(get(ITh,'UserData'))  % The user might have closed the figure with the current image handle, check that it exists!
    switch action
        case {'NewWindow'}        % Show image in a new window
            drawnow
            figure
            data = get(get(ITh,'UserData'),'Cdata');
            if ndims(data) == 2
                imagesc(data),axis image,colormap gray    % Scalar image
            else
                image(data),axis image                    % RGB image
            end
            title(get(get(ITh,'UserData'),'Tag'))
        case {'Histogram'}                                % Produce histogram (only for scalar images)
            drawnow
            figure
            data = get(get(ITh,'UserData'),'Cdata');
            hist(data(:),min(200,round(length(data(:))/150)));
            title(['Histogram for ' get(get(ITh,'UserData'),'Tag')])
            grid on
        case {'MatlabWS'}                                 % Store image in Matlab base work space
            assignin('base','ImageToolIm',get(get(ITh,'UserData'),'Cdata'));
        otherwise
            disp('Unknown action')                        % Should never get here, but just in case.
    end
end