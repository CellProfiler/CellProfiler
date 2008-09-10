function img = CPimcapture( h, opt, dpi, opt2, opt3)
% IMCAPTURE do screen captures at controllable resolution using the 
%   undocumented "hardcopy" built-in function.
%
% USAGE:
%   IMG = IMCAPTURE(H) gets a screen capture from object H, where H is a 
%                   handle to a figure, axis or an image
%                   When H is an axis handle and OPT ~= 'all' (see below) 
%                   the capture is done on that axis alone, no matter how 
%                   many more axis the Figure contains. 
%   IMG = IMCAPTURE or IMG = IMCAPTURE([]) operates on the current figure.
%   IMG = IMCAPTURE(H, OPT) selects one of three possible operations 
%                   depending on the OPT string.
%           OPT == 'img' returns an image corresponding to the axes 
%                   contents and a size determined by the figure size.
%           OPT == 'imgAx' returns an image corresponding to the displyed 
%                   image plus its labels. This option tries to fill a 
%                   enormous gap in Matlab base which is the non 
%                   availability of an easy way of geting a raster image 
%                   that respects the data original aspect ratio.
%           OPT == 'all' returns an image with all the figure's contents 
%                   and size determined by the figure size.
%   IMG = IMCAPTURE(H,OPT,DPI) do the screen capture at DPI resolution. 
%                   DPI can be either a string or numeric. If DPI is not 
%                   provided, the screen capture is done at 150 dpi. If 
%                   DPI = 0, returns an image that has exactly the same 
%                   width and height of the original CData. You can use 
%                   this, for example, to convert an indexed image to RGB.
%                   Note, this should be equivalent to a call to GETFRAME, 
%                   but maybe it fails less in returning EXACTLY an image 
%                   of the same size of CData (R13, as usual, is better 
%                   than R14).
%   IMG = IMCAPTURE(H,'img',[MROWS NCOLS]) returns an image of the size 
%                   specified by [mrows ncols].
%                   This a 10 times faster option to the use of 
%                   imresize(IMG,[mrows ncols],method) when method is 
%                   either 'bilinear' or 'bicubic'. Not to mention the 
%                   memory consumption.
%
%   Written by Joaquim Luis (jluis@ualg.pt)
%
%   Please see the documentation at http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=13355
%   for the full explnation of the code. 

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

hFig = [];      hAxes = [];
if (nargin == 0 || isempty(h)),     h = get(0,'CurrentFigure');    end
if (~ishandle(h))
    error('imcapture:a','First argument is not a valid handle')
end
if (strcmp(get(h,'Type'),'figure'))
    hAxes = findobj(h,'Type','axes');
elseif (strcmp(get(h,'Type'),'axes'))
    hAxes = h;
    h = get(h,'Parent');
elseif (strcmp(get(h,'Type'),'image'))
    hAxes = get(h,'Parent');
    h = get(hAxes,'Parent');
else
    h = [];
end
if (~ishandle(h))
    error('imcapture:a','First argument is not a valid Fig/Axes/Image handle')
end
if (nargin <= 1),   opt = [];   end

inputargs{1} = h;
renderer = get(h, 'Renderer');

if (nargout == 1)                   % Raster mode. We expect to return a RGB array
    inputargs{4} = '-r150';         % Default value for the case we got none in input
    inputargs{2} = 'lixo.jpg';      % The name doesn't really matter, but we need one.
    if strcmp(renderer,'painters')
        renderer = 'zbuffer';
    end
    inputargs{3} = ['-d' renderer];
    if (nargin == 3)
        if (isnumeric(dpi) && numel(dpi) == 1)
            inputargs{4} = ['-r' sprintf('%d',dpi)];
        elseif (isnumeric(dpi) && numel(dpi) == 2)      % New image size in [mrows ncols]
            inputargs{4} = dpi;
        elseif (ischar(dpi))
            inputargs{4} = ['-r' dpi];
        else
            error('imcapture:a','third argument must be a ONE or TWO elements vector, OR a char string')
        end
    end
end

msg = [];
if (numel(hAxes) == 1 && strcmp( get(hAxes,'Visible'),'off') && (nargin == 1 || isempty(opt)) )
    % Default to 'imgOnly' when we have only one image with invisible axes
    opt = 'img';
end

if (nargin > 1)
    switch opt
        case 'img'      % Capture the image only
            [img,msg] = imgOnly([],hAxes,inputargs{:});
        case 'imgAx'    % Capture an image including the Labels
            [img,msg] = imgOnly('nikles',hAxes,inputargs{:});
        case 'all'      % Capture everything in Figure
            img = allInFig(inputargs{:});
        otherwise
            msg = 'Second argument is not a recognized option';
    end
else
    img = allInFig(inputargs{:});
end

if (~isempty(msg))      % If we had an error inside imgOnly()
    error('imcapture:a',msg);        img = [];
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%
% SUBFUNCTION: imgOnly %
%%%%%%%%%%%%%%%%%%%%%%%%
function [img, msg] = imgOnly(opt, hAxes, varargin)
% Capture the image, and optionaly the frame, mantaining the original image aspect ratio.
% We do that be messing with the Figure's 'PaperPosition' property
h = varargin{1};    msg = [];
if (isempty(hAxes) || numel(hAxes) > 1)
    msg = 'With the selected options the figure must contain one, and one ONLY axes';
    return
end
im = get(findobj(h,'Type','image'),'CData');
if (~isempty(im))
    nx = size(im,2);                ny = size(im,1);
else                    % We have something else. A plot, a surface, etc ...
    axUnit = get(hAxes,'Units');    set(hAxes,'Units','pixels')
    axPos = get(hAxes,'pos');       set(hAxes,'Units',axUnit)
    nx = axPos(3);                  ny = axPos(4);
end

PU = get(h,'paperunits');       set(h,'paperunits','inch')
pp = get(h,'paperposition');    PPM = get(h,'PaperPositionMode');
dpi = (nx / pp(3));             % I used to have a round() here but this pobably more correct
% Here is the key point of all this manip.
set(h,'paperposition',[pp(1:3) ny / dpi])

axUnit = get(hAxes,'Units');
axPos = get(hAxes,'pos');           % Save this because we will have to restore it later
set(hAxes,'Units','Normalized')     % This is the default, but be sure
fig_c = get(h,'Color');

all_axes = findobj(h,'Type','axes');
% If there are more than one axis, put them out of sight so that they wont interfere
% (Just make them invisible is not enough)
if (numel(all_axes) > 1)
    other_axes = setxor(hAxes,all_axes);
    otherPos = get(other_axes,'Pos');
    bakAxPos = otherPos;
    if (~iscell(otherPos))
        otherPos(1) = 5;
        set(other_axes,'Pos',otherPos)
    else
        for (i=1:numel(other_axes))
            otherPos{i}(1) = 5;
            set(other_axes(i),'Pos',otherPos{i})
        end
    end
end

have_frames = false;
axVis = get(hAxes,'Visible');
if (isempty(opt))                   % Pure Image only capture. Even if axes are visible, ignore them
    set(hAxes,'pos',[0 0 1 1],'Visible','off')
elseif (strcmp(get(hAxes,'Visible'),'on'))  % Try to capture an image that respects the data aspect ratio
    have_frames = true;
    h_Xlabel = get(hAxes,'Xlabel');         h_Ylabel = get(hAxes,'Ylabel');
    units_save = get(h_Xlabel,'units');
    set(h_Xlabel,'units','pixels');         set(h_Ylabel,'units','pixels');
    Xlabel_pos = get(h_Xlabel,'pos');
    Ylabel_pos = get(h_Ylabel,'Extent');

    if (abs(Ylabel_pos(1)) < 30)    % Stupid hack, but there is a bug somewhere
        Ylabel_pos(1) = 30;
    end

    y_margin = abs(Xlabel_pos(2))+get(h_Xlabel,'Margin');  % To hold the Xlabel height
    x_margin = abs(Ylabel_pos(1))+get(h_Ylabel,'Margin');  % To hold the Ylabel width
    y_margin = min(max(y_margin,20),30);            % Another hack due to the LabelPos non-sense

    figUnit = get(h,'Units');        set(h,'Units','pixels')
    figPos = get(h,'pos');           set(h,'Units',figUnit)
    x0 = x_margin / figPos(3);
    y0 = y_margin / figPos(4);
    set(hAxes,'pos',[x0 y0 1-[x0 y0]-1e-2])
    set(h_Xlabel,'units',units_save);     set(h_Ylabel,'units',units_save);
else            % Dumb choice. 'imgAx' selected but axes are invisible. Default to Image only
    set(hAxes,'pos',[0 0 1 1],'Visible','off')
end

confirm = false;
try
    if (strcmp(varargin{4},'-r0'))              % One-to-one capture
        varargin{4} = ['-r' sprintf('%d',round(dpi))];
        confirm = true;
        mrows = ny;            ncols = nx;      % To use in "confirm"
    elseif (numel(varargin{4}) == 2)            % New size in mrows ncols
        mrows = varargin{4}(1);
        ncols = varargin{4}(2);
        if (~have_frames)
            set(h,'paperposition',[pp(1:2) ncols/dpi mrows/dpi])
            confirm = true;
        else                        % This is kind of very idiot selection, but let it go
            wdt = pp(3);          hgt = pp(3) * mrows/ncols;
            set(h,'paperposition',[pp(1:2) wdt hgt])
        end
        varargin{4} = ['-r' sprintf('%d',round(dpi))];
    else                            % Fourth arg contains the dpi
        if (have_frames)
            wdt = pp(3);          hgt = pp(3) * ny/nx;
            set(h,'paperposition',[pp(1:2) wdt hgt])
        end
    end
    if (numel(varargin) == 5)       % Vector graphics formats
        set(h,'paperposition',[pp(1:2) varargin{5}*2.54])       % I warned in doc to use CM dimensions
        varargin(5) = [];           % We don't want it to go into hardcopy
    end

    img = hardcopy( varargin{:} );      % Capture
    if (confirm)                        % We asked for a pre-determined size. Check that the result is correct
        dy = mrows - size(img,1);       % DX & DY should be zero or one (when it buggs).
        dx = ncols - size(img,2);
        if (dx ~= 0 || dy ~= 0)         % ML failed (probably R14). Repeat to make it obey
            mrows_desBUG = mrows + dy;
            ncols_desBUG = ncols + dx;
            set(h,'paperposition',[pp(1:2) ncols_desBUG/dpi mrows_desBUG/dpi])
            img = hardcopy( varargin{:} );      % Insist
        end
    end
catch                                   % If it screws, restore original Fig properties anyway
    set(hAxes,'Units',axUnit,'pos',axPos,'Visible','on')
    set(h,'paperposition',pp,'paperunits',PU,'PaperPositionMode',PPM,'Color',fig_c)
    msg = lasterr;      img = [];
end

% If there are more than one axis, bring them to their's original positions
if (numel(all_axes) > 1)
    if (~iscell(otherPos))
        set(other_axes,'Pos',bakAxPos)
    else
        for (i=1:numel(other_axes)),    set(other_axes(i),'Pos',bakAxPos{i});  end
    end
end

% Reset the original fig properties
set(hAxes,'Units',axUnit,'pos',axPos,'Visible',axVis)
set(h,'paperposition',pp,'paperunits',PU,'PaperPositionMode',PPM,'Color',fig_c)
    
%%
%%%%%%%%%%%%%%%%%%%%%%%%%
% SUBFUNCTION: allInFig 
%%%%%%%%%%%%%%%%%%%%%%%%%
function img = allInFig(varargin)

% Get everything in the Figure
h = varargin{1};
fig_c = get(h,'Color');
if (numel(varargin) == 3)
    varargin{4} = '-r150';
end
img = hardcopy( varargin{:} );    
set(h,'Color',fig_c)
