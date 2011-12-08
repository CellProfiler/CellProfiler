"""client.py The subimager client

"""
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2011
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org
#

import atexit
import os
import httplib
import threading
import subprocess
import urllib
import sys
import numpy as np
import logging

if __name__ == "__main__":
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(logging.StreamHandler())
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    
logger = logging.getLogger(__name__)
port = 0
init_semaphore = threading.Semaphore(0)
stop_semaphore = threading.Semaphore(0)
subprocess_thread = None
logger_thread = None
subimager_running = False
subimager_process = None
conn = None
if hasattr(sys, 'frozen'):
    __root_path = os.path.split(os.path.abspath(sys.argv[0]))[0]
else:
    __root_path = os.path.abspath(os.path.split(__file__)[0])
    __root_path = os.path.split(__root_path)[0]
__jar_path = os.path.join(__root_path, 'subimager', 'subimager.jar')

def start_subimager():
    '''Start the subimager subprocess if it is not yet running'''
    global init_semaphore, stop_semaphore, subprocess_thread, port, conn
    global __jar_path
    if subimager_running:
        return
    
    subprocess_thread = threading.Thread(target = run_subimager)
    subprocess_thread.setDaemon(True)
    subprocess_thread.start()
    init_semaphore.acquire()
    conn = httplib.HTTPConnection("localhost", port,
                              source_address=("127.0.0.1", 0))

def run_subimager():
    '''Thread function for controlling the subimager process'''
    
    global port, init_semaphore, stop_semaphore, subimager_process
    global subimager_running
    
    logging.debug("Starting subimager subprocess")
    args = [
        "java",
        "-cp",
        __jar_path,
        "org.cellprofiler.subimager.Main"]
    log4j_file = os.path.join(__root_path, "subimager", "log4j.properties")
    if os.path.exists(log4j_file):
        log4j_file = "file:"+urllib.pathname2url(log4j_file)
        args = args[:1] + ["-Dlog4j.configuration=%s" % log4j_file] + args[1:]
    subimager_process = subprocess.Popen(args, 
                         stdout=subprocess.PIPE)
    def on_exit(subimager_process = subimager_process):
        if subimager_process != None:
            subimager_process.terminate()
            
    atexit.register(on_exit)
    port = int(subimager_process.stdout.readline().strip())
    logging.debug("Subimager subprocess started on port %d" % port)
    subimager_running = True
    stdout_thread = threading.Thread(target=run_logger)
    stdout_thread.setDaemon(True)
    stdout_thread.start()
    init_semaphore.release()
    stop_semaphore.acquire()
    stdoutdata, stderrdata = subimager_process.communicate()
    print stdoutdata
    subimager_running = False

def run_logger():
    global subimager_process
    while(True):
        try:
            logger.info(subimager_process.stdout.readline().strip())
        except:
            break
        
def get_image(url, **kwargs):
    '''Get an image via the readimage web interface
    
    url - file or http url to be used to fetch the image
    
    Accepts web query parameters via keyword arguments, for example,
    get_image(path, groupfiles=True, allowfileopen=True, series=3, index=1, channel=2)
    '''
    global conn
    query_params = [("url", url)]
    query_params += list(kwargs.items())
    query = urllib.urlencode(query_params)
    url = "/readimage?%s" % query
    
    conn.request("GET", url)
    response = conn.getresponse(buffering=True)
    if response.status != httplib.OK:
        logger.warn(response.reason)
        logger.warn(response.read())
        raise RuntimeError("Image server failed to read image. URL="+url)
    cookie = unicode(response.read(8), "utf-8")
    version = np.ndarray(shape=(1,), dtype=">u4", buffer=response.read(4))[0]
    ndim, width, height = np.ndarray(shape=(3,), dtype=">u4", buffer=response.read(12))
    if (ndim == 3):
        channels = np.ndarray(shape=(1,), dtype=">u4", buffer=response.read(4))[0]
        image = np.ndarray(shape = (channels, height, width), dtype=">u2", 
                           buffer = response.read(width * height * channels * 2))
        image = image.transpose(1,2,0)
    else:
        image = np.ndarray(shape = (height, width), dtype=">u2", 
                           buffer = response.read(width * height * 2))
    response.close()
    return image

def get_metadata(url, **kwargs):
    '''Get metadata for an image file
    url - file or http url to be used to fetch the image
    
    Accepts web query parameters via keyword arguments, for example,
    get_image(path, groupfiles=True, allowfileopen=True, series=3, index=1, channel=2)
    '''
    global conn
    query_params = [("url", url)]
    query_params += list(kwargs.items())
    query = urllib.urlencode(query_params)
    url = "/readmetadata?%s" % query
    
    conn.request("GET", url)
    response = conn.getresponse(buffering=True)
    return unicode(response.read(), 'utf-8')

def stop_subimager():
    '''Stop the subimager process by web command'''
    global conn
    conn.request("GET", "/stop")
    conn.getresponse()
    stop_semaphore.release()
    
if __name__ == "__main__":
    import wx
    import matplotlib
    matplotlib.use("WX")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg, FigureManager
    sys.path.append(__root_path)
    from external_dependencies import fetch_external_dependencies
    
    fetch_external_dependencies()
    start_subimager()
    app = wx.PySimpleApp()
    frame = wx.Frame(None,size=(1024,768))
    menu = wx.Menu()
    metadata_item_id = wx.NewId()
    menu.Append(wx.ID_OPEN, "Open")
    menu.Append(metadata_item_id, "Open metadata")
    menu.Append(wx.ID_EXIT, "Exit")
    menubar = wx.MenuBar()
    menubar.Append(menu, "File")
    frame.SetMenuBar(menubar)
    frame.Sizer = wx.BoxSizer()
    figure = Figure()
    canvas = FigureCanvasWxAgg(frame, -1, figure)
    frame.Sizer.Add(canvas, 1, wx.EXPAND | wx.ALL, 3)
    figure_manager = FigureManager(canvas, 1, frame)
    
    def on_open_image(event, figure=figure, frame = frame, canvas = canvas):
        dialog = wx.FileDialog(frame)
        dialog.Title = "Choose an image"
        dialog.Wildcard = "JPeg file (*.jpg)|*.jpg|PNG file (*.png)|*.png|Tiff file (*.tif)|*.tif|Flex file (*.flex)|*.flex|Any file (*.*)|*.*"
        if dialog.ShowModal() == wx.ID_OK:
            image = get_image("file:" + urllib.pathname2url(dialog.Path))
            image = image.astype(float) / 65535
            figure.clf()
            axes = figure.add_subplot(1,1,1)
            if image.ndim == 2:
                axes.imshow(image, matplotlib.cm.gray)
            else:
                axes.imshow(image)
            canvas.draw()
        dialog.Destroy()
    frame.Bind(wx.EVT_MENU, on_open_image, id=wx.ID_OPEN)
    
    def on_open_metadata(event, frame = frame):
        dialog = wx.FileDialog(frame)
        dialog.Title = "Choose an image"
        dialog.Wildcard = "JPeg file (*.jpg)|*.jpg|PNG file (*.png)|*.png|Tiff file (*.tif)|*.tif|Flex file (*.flex)|*.flex|Any file (*.*)|*.*"
        if dialog.ShowModal() == wx.ID_OK:
            metadata = get_metadata("file:" + urllib.pathname2url(dialog.Path))
            savedlg = wx.FileDialog(frame, style= wx.FD_SAVE)
            savedlg.Title = "Save metadata"
            savedlg.Wildcard = "XML file (*.xml)}*.xml"
            if savedlg.ShowModal() == wx.ID_OK:
                fd = open(savedlg.Path, "w")
                fd.write(metadata)
            savedlg.Destroy()
        dialog.Destroy()
    frame.Bind(wx.EVT_MENU, on_open_metadata, id=metadata_item_id)
        
    def on_exit(event, frame=frame):
        frame.Close()
    frame.Bind(wx.EVT_MENU, on_exit, id=wx.ID_EXIT)
    frame.Layout()
    frame.Show()
    app.MainLoop()
    stop_subimager()
    subprocess_thread.join()

