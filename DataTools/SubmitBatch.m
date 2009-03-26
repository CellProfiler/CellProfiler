function handles=SubmitBatch(handles)

% Help for the Submit Batch tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Submits batches made by the CreateBatchFiles module to the cluster 
% via webserver.
% ************************************************************************
% This tool makes a webserver call to the URL,
% "http://imageweb.broad.mit.edu/batchprofiler/cgi-bin/NewBatch.py", to
% submit the batch files created by the CreateBatchFiles module to the
% cluster using CPCluster.py and bsub. The webserver creates a record
% in batchprofiler.batch for the batch, documenting the files and
% creates one record in batchprofiler.run per bsub job submission.
%
% The tool's UI collects the following fields:
% data_dir   - the directory that holds "Batch_data.mat" which holds the
%              details for running the batch using CPCluster.py
% email      - the email of the submitter. The webserver sends a brief
%              email which gives the submitter the batch ID (which is
%              the primary key for the batch table) as a link to a
%              webpage that lets the user monitor the job's progress.
% queue      -  one of the bsub queues. see
% http://iwww.broad.mit.edu/itsystems/lsf_clusters.html#whatqueues
%              for details.
% write_data - whether or not to write the Batch_##_to_##_OUT.mat files.
% batch_size - # of image sets per bsub submission.
% cpcluster  - the revision # of the version of CPCluster to use. This
%              should correspond to the #### part of a directory like
%              /imaging/analysis/CPCluster/####.
% timeout    - the timeout parameter to CPCluster: how long to let a job
%              run before timing it out.
%
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
    info=ObjectsToSubmit(handles);
    if info.Userdata == 1
        write_data='N';
        if info.WriteData== 1
            write_data='Y';
        end;
        url=sprintf('http://%s/batchprofiler/cgi-bin/NewBatch.py?email=%s&queue=%s&data_dir=%s&write_data=%s&batch_size=%d&timeout=%f&cpcluster=%s',...
            info.Server,info.Email,info.Queue,info.DataDir,write_data,info.BatchSize,info.Timeout,info.CPCluster);
        result=urlread(url);
        web(strcat('text://',result));
    end;
end

function hText=uitext(h,text, position)
    GUIhandles = guidata(gcbo);
    FontSize = GUIhandles.Preferences.FontSize;
    hText=uicontrol(h,'style','text','String',text,'FontName','helvetica',...
        'FontSize',FontSize,'units','pixels','position',position,...
        'BackgroundColor',get(h,'color'),'fontweight','bold');
end

function hEdit=uiedit(h,default_text,position)
    GUIhandles = guidata(gcbo);
    FontSize = GUIhandles.Preferences.FontSize;
    hEdit=uicontrol(h,'Style','edit','units','pixels','position',position,...
        'backgroundcolor',[1 1 1],'String',default_text,'FontSize',FontSize);
end

function hButton=uipushbutton(h,text,position,callback)
    GUIhandles = guidata(gcbo);
    FontSize = GUIhandles.Preferences.FontSize;
    hButton=uicontrol(h,'style','pushbutton','String',text,'FontName','helvetica','FontSize',FontSize,'FontWeight', 'bold','units','pixels',...
    'position',position,'Callback',callback,'BackgroundColor',[.7 .7 .9]);
end

function hBrowseButton = uibrowsedir(h,textctl, dlgcaption, position)
    callback=@(hObject,eventdata) browsedir_cb(textctl,dlgcaption);
    hBrowseButton=uipushbutton(h,'Browse...',position, callback);
end

function resultdir=browsedir_cb(textctl,dlgcaption)
    CurrentChoice = get(textctl,'string');
    if exist(CurrentChoice, 'dir')
        tempdir = CurrentChoice;
    else
        tempdir=pwd;
    end;
    resultdir = CPuigetdir(tempdir,dlgcaption);
    if resultdir ~= 0 
        set(textctl,'String', resultdir);
    end;
end

function SubmitInfo=ObjectsToSubmit(handles)
SubmitInfo.Userdata = 0;
SubmitInfo.DataDir = handles.Current.DefaultOutputDirectory;
SubmitInfo.UnixPath = '/imaging/analysis';
if strcmp(computer,'PCWIN') || strcmp(computer,'PCWIN64')
    SubmitInfo.LocalPath = '\\iodine\imaging_analysis';
else
    SubmitInfo.LocalPath = SubmitInfo.UnixPath;
end;
SubmitInfo.Email = [];
SubmitInfo.Queue = 'broad';
SubmitInfo.Server = 'imageweb.broad.mit.edu';
try
    if isdeployed
        svn_ver_char = handles.Settings.CurrentSVNVersion;
    else
        svn_ver_char = CPsvnversionnumber(handles.Preferences.DefaultModuleDirectory);
    end
catch
    svn_ver_char = '0';
end
SubmitInfo.CPCluster = num2str(svn_ver_char);

SubmitInfo.BatchSize = 10;
SubmitInfo.Timeout = 30;
SubmitInfo.WriteData = 1;

% Create Submit window
SBh = CPfigure;
set(SBh,'UserData',0);
set(SBh,'units','pixels','resize','on','menubar','none','toolbar','none','numbertitle','off',...
    'Name','Submit batch window','CloseRequestFcn','set(gcf,''UserData'',0);uiresume(gcbf)');
[ScreenWidth,ScreenHeight] = CPscreensize;
Height = 480;
Width  = 600;
GUIhandles = guidata(gcbo);
FontSize = GUIhandles.Preferences.FontSize;
FontSizePixels = FontSize * get(0,'ScreenPixelsPerInch') / 72;
uiheight = FontSizePixels*2;
editheight = FontSizePixels * 5 / 3;
textheight = FontSizePixels * 4 / 3;
gutter = 10;
colwidth = [120 350 Width-120-350-gutter*3];
colstart = [0 colwidth(1)+gutter colwidth(1)+colwidth(2)+gutter*2];

LeftPos = (ScreenWidth-Width)/2;
BottomPos = (ScreenHeight-Height)/2;
set(SBh,'position',[LeftPos BottomPos Width Height]);
%
% Data directory
%
line = Height-uiheight-5;
uitext(SBh,'Data directory',[colstart(1) line colwidth(1) textheight]);
DataDirCtl=uiedit(SBh,SubmitInfo.DataDir,[colstart(2) line colwidth(2) editheight]);
uibrowsedir(SBh,DataDirCtl,'Select the data directory',[colstart(3) line colwidth(3) uiheight-2]);
%
% Local path
%
line = line - uiheight;
uitext (SBh,'Local mountpoint',[colstart(1) line colwidth(1) textheight]);
LocalPathCtl=uiedit(SBh,SubmitInfo.LocalPath,[colstart(2) line colwidth(2) editheight]);
uibrowsedir(SBh,LocalPathCtl,'Select the imaging mountpoint for your machine',...
    [colstart(3) line colwidth(3) uiheight-2]);
%
% Unix path
%
line = line - uiheight;
uitext(SBh, 'Unix mountpoint',[colstart(1) line colwidth(1) textheight]);
UnixPathCtl=uiedit(SBh,SubmitInfo.UnixPath,[colstart(2) line colwidth(2) editheight]);
%
% Email
%
line = line - uiheight;
uitext(SBh,'e-mail',[colstart(1) line colwidth(1) textheight]);
EmailCtl=uiedit(SBh,SubmitInfo.Email,[colstart(2) line 200 editheight]);
%
% Queue
%
line = line - uiheight;
uitext(SBh, 'Queue',[colstart(1) line colwidth(1) textheight]);
QueueCtl=uiedit(SBh, SubmitInfo.Queue,[colstart(2) line 200 editheight]);
%
% Server
%
line = line -  uiheight;
uitext(SBh, 'Server',[colstart(1) line colwidth(1) textheight]);
ServerCtl=uiedit(SBh, SubmitInfo.Server,[colstart(2) line 200 editheight]);
%
% CPCluster version #
%
line = line - uiheight;
uitext(SBh, 'CPCluster version', [colstart(1) line colwidth(1) textheight]);
CPClusterCtl=uiedit(SBh, SubmitInfo.CPCluster, [colstart(2) line 100 editheight]);
%
% Batch size
%
line = line - uiheight;
uitext(SBh, 'Batch size',[colstart(1) line colwidth(1) textheight]);
BatchSizeCtl=uiedit(SBh, SubmitInfo.BatchSize,[colstart(2) line 100 editheight]);
%
% Timeout
%
line = line - uiheight;
uitext(SBh, 'Timeout', [colstart(1) line colwidth(1) textheight]);
TimeoutCtl=uiedit(SBh, SubmitInfo.Timeout,[colstart(2) line 100 editheight]);
%
% write data
%
line = line - uiheight;
WriteDataCtl = uicontrol(...
    'Style','checkbox',...
    'String','Write data',...
    'FontSize', FontSize,...
    'FontName', 'helvetica',...
    'Value',SubmitInfo.WriteData,...
    'Position',[colstart(2) line colwidth(2) textheight]);
%
% Cancel and Submit buttons
%
posx = (Width - 200)/2;               % Centers buttons horizontally
uipushbutton(SBh,'Cancel',[posx 10 75 uiheight],'close(gcf)');
uipushbutton(SBh,'Submit',[posx+125 10 75 uiheight],'[foo,fig] = gcbo;set(fig,''UserData'',1);uiresume(gcbf);clear fig foo');
uiwait()
if get(SBh,'Userdata') == 1,     % The user pressed the Submit button
    CPmsgbox(['PLEASE WAIT while the webserver submits the batches for processing.  ' ...
        'A window should appear with your batch number, however this may take a minute or so...']);
    unix_dir = get(DataDirCtl,'String');
    local_path = get(LocalPathCtl, 'String');
    unix_path = get(UnixPathCtl, 'String');
    unix_dir = strrep(unix_dir,local_path,unix_path);
    unix_dir = strrep(unix_dir,'\','/');
    SubmitInfo.Userdata = 1;
    SubmitInfo.DataDir = unix_dir;
    SubmitInfo.Email = get(EmailCtl,'String');
    SubmitInfo.Queue = get(QueueCtl,'String');
    SubmitInfo.Server = get(ServerCtl,'String');
    SubmitInfo.Timeout = str2double(get(TimeoutCtl,'String'));
    SubmitInfo.BatchSize = str2double(get(BatchSizeCtl,'String'));
    SubmitInfo.CPCluster = get(CPClusterCtl,'String');
    SubmitInfo.WriteData = get(WriteDataCtl,'Value');
end
delete(SBh)
end