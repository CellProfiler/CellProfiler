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

from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
import httplib
import logging
import numpy as np
import re
import socket
import subprocess
import sys
import threading
import urllib
import zlib

if hasattr(sys, 'frozen'):
    __root_path = os.path.split(os.path.abspath(sys.argv[0]))[0]
else:
    __root_path = os.path.abspath(os.path.split(__file__)[0])
    __root_path = os.path.split(__root_path)[0]
    
if __name__ == "__main__":
    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(logging.StreamHandler())
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    sys.path.append(__root_path)
from cellprofiler.cpmath.filter import paeth_decoder
    
logger = logging.getLogger(__name__)
port = 0
init_semaphore = threading.Semaphore(1)
stop_semaphore = threading.Semaphore(0)
subprocess_thread = None
logger_thread = None
subimager_running = False
subimager_process = None
subimager_deadman_socket = None
subimager_deadman_connection = None

__jar_path = os.path.join(__root_path, 'subimager', 'subimager.jar')

def connect():
    return httplib.HTTPConnection("localhost", port,
                                  source_address=("127.0.0.1", 0))

###############################
#
# About the subimager process:
#
#    The process is started in the subprocess thread. A pair of semaphores
#    signal initiation and termination:

#    init_semaphore - Acquired before starting subprocess to ensure the
#                     subprocess thread is only started once.  Afterward, the
#                     subprocess thread signals the main thread that it has
#                     started by releasing this semaphore. The client knows the
#                     HTTP server at this point and has established the deadman
#                     server socket which, when closed, will cause the child
#                     process to terminate.
#
#     stop_semaphore - any thread can shut down the subprocess thread by
#                      releasing the stop_semaphore. The subprocess thread
#                      will close the deadman server socket which will
#                      cause the child process to terminate. 
#
#     The logging thread monitors the subimager process's STDOUT and logs
#     whatever text comes from the process.
#
##############################
def start_subimager():
    '''Start the subimager subprocess if it is not yet running'''
    global subprocess_thread

    init_semaphore.acquire()  # make sure subprocess is only started once...
    if subimager_running:
        init_semaphore.release()
        return

    subprocess_thread = threading.Thread(target = run_subimager)
    subprocess_thread.setDaemon(True)
    subprocess_thread.start()
    init_semaphore.acquire()  # wait for subprocess thread to release
    init_semaphore.release()

def run_subimager():
    '''Thread function for controlling the subimager process'''
    
    global port, init_semaphore, stop_semaphore, subimager_process
    global subimager_running, subimager_deadman_socket, subimager_deadman_connection

    subimager_deadman_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    subimager_deadman_socket.bind(("127.0.0.1", 0))
    subimager_deadman_socket.listen(1)
    deadman_addr, deadman_port = subimager_deadman_socket.getsockname()
    logging.debug("Deadman port = %d" % deadman_port)
    logging.debug("Starting subimager subprocess")
    args = [
        "java",
        "-cp",
        __jar_path,
        "org.cellprofiler.subimager.Main",
        "--deadman-server=%s:%d" % (deadman_addr, deadman_port)
    ]
    log4j_file = os.path.join(__root_path, "subimager", "log4j.properties")
    if os.path.exists(log4j_file):
        log4j_file = "file:"+urllib.pathname2url(log4j_file)
        args = args[:1] + ["-Dlog4j.configuration=%s" % log4j_file] + args[1:]
    subimager_process = subprocess.Popen(args, 
                         stdout=subprocess.PIPE)
    port = int(subimager_process.stdout.readline().strip())
    logging.debug("Subimager subprocess started on port %d" % port)
    (subimager_deadman_connection, client_addr) = subimager_deadman_socket.accept()
    logging.debug("Connected to deadman at %s" % str(client_addr))
    subimager_running = True
    stdout_thread = threading.Thread(target=run_logger)
    stdout_thread.setDaemon(True)
    stdout_thread.start()
    init_semaphore.release()
    stop_semaphore.acquire()
    subimager_deadman_connection.close()
    subimager_process.wait()  # output is handled by the run_logger thread
    subimager_running = False

def run_logger():
    while(True):
        try:
            logger.info(subimager_process.stdout.readline().strip())
        except:
            break

class HTTPError(RuntimeError):
    '''An encapsulation of an HTTP error code
    
    self.status - the HTTP status code (e.g. 401 = url not found)
    self.context - context message about the call, for instance that the
                   server failed to read the image
    self.body - the body of the HTTP response
    '''
    def __init__(self, response, context):
        assert isinstance(response, httplib.HTTPResponse)
        self.status = response.status
        self.context = context
        logger.info("%d: %s. %s" % (self.status, response.reason,
                                    self.context))
        try:
            self.body = response.read()
            logger.debug(self.body)
        except:
            self.body = None
            pass
        
def get_image(url, **kwargs):
    '''Get an image via the readimage web interface
    
    url - file or http url to be used to fetch the image
    
    Accepts web query parameters via keyword arguments, for example,
    get_image(path, groupfiles=True, allowfileopen=True, series=3, index=1, channel=2)
    '''
    query_params = [("url", url)]
    query_params += list(kwargs.items())
    query = urllib.urlencode(query_params)
    url = "/readimage?%s" % query
    
    conn = connect()
    try:
        conn.request("GET", url, headers={"Connection":"close"})
        response = conn.getresponse(buffering=True)
        if response.status != httplib.OK:
            raise HTTPError(response, "Image server failed to read image. URL="+url)
        return decode_image(response.read())
    finally:
        conn.close()

def post_image(url, image, omexml, **kwargs):
    '''Post an image to the webserver via the writeimage web interface
    
    url - file or http url of the file to be created
    image - an N-D image, scaled appropriately for the OME data type
    omexml - the OME xml metadata for the image
    
    Accepts post parameters via keyword arguments, for example,
    post_image(path, compression="LZW", series=3, index=1, channel=2)
    '''
    
    message = MIMEMultipart()
    message.set_type("multipart/form-data")
    message.add_header("Connection","close")
    d = dict([(key, MIMEText(value.encode("utf-8"), "plain", "utf-8"))
              for key, value in kwargs.iteritems()])
    d["omexml"] = MIMEText(omexml.encode("utf-8"), "xml", "utf-8")
    d["image"] = MIMEApplication(encode_image(image))
    d["url"] = MIMEText(url)
    for key, value in d.iteritems():
        value.add_header("Content-Disposition", "form-data", name=key)
        message.attach(value)
    
    conn = connect()
    try:
        body = message.as_string()
        # oooo - translate line-feeds w/o carriage returns into cr/lf
        #        The generator uses print to write the message, how bad.
        #        This keeps us from being able to encode in binary too.
        #
        body = "\r\n".join(re.split("(?<!\\r)\\n", body))
        
        conn.request("POST", "/writeimage", body, dict(message.items()))
        response = conn.getresponse()
        if response.status != httplib.OK:
            raise HTTPError(response, "Image server failed to write image. URL="+url)
    finally:
        conn.close()

def get_metadata(url, **kwargs):
    '''Get metadata for an image file
    url - file or http url to be used to fetch the image
    
    Accepts web query parameters via keyword arguments, for example,
    get_image(path, groupfiles=True, allowfileopen=True, series=3, index=1, channel=2)
    '''
    query_params = [("url", url)]
    query_params += list(kwargs.items())
    query = urllib.urlencode(query_params)
    url = "/readmetadata?%s" % query
    
    conn = connect()
    try:
        conn.request("GET", url, headers={"Connection":"close"})
        response = conn.getresponse(buffering=True)
        if response.status != httplib.OK:
            raise HTTPError(response, "Image server failed to get metadata. URL="+url)
        return unicode(response.read(), 'utf-8')
    finally:
        conn.close()

def encode_image(a):
    '''Encode the array according to the subimager protocol
    
    a - array in question. Color images should be interleaved (color
        dimension is last). Values should be floating point and should be
        scaled appropriately for the pixel type being saved.
       
    The compression algorithm:
  
    * Use a Paeth filter as described in http://www.w3.org/TR/PNG-Filters.html
      to encode pixels in each plane as differences to pixels previously transmitted.
      The Paeth filter has as it's inputs the following bytes:
     
      C[8] B[8]
      A[8] x[8]
     
      where the double is decomposed into it's 8-byte IEEE representation
      in network (high-endian) byte order
     
      The algorithm is
      * compute P = A[i] + B[i] - C[i]
      * if abs(P-A[i]) < abs(P-B[i]) and abs(P-A[i]) < abs(P-C[i]) transmit X[i] - A[i]
      * else if abs(P-B[i]) < abs(P-C[i]) transmit X[i] - B[i] 
      * else transmit X[i] - C[i]
      *
      * A, B and C are taken to be zero for the corner case.
    
    * Compress the stream using DEFLATE
 
    The header has the following format:
    
    0 - 7: Cookie = "CPNDIMAGE"
    8 - 11: version number, unsigned integer, currently = 1 
    12 - 15: NDIM = # of dimensions, unsigned integer
    16 - 16 + 4*NDIM: the sizes of each of the dimensions
    '''
    ndim = a.ndim
    shape = a.shape
    x = np.asanyarray(a, dtype=">f8")
    data = x.tostring()
    x = np.frombuffer(data, "u1").copy()
    #
    # Organize the data as rasters of possibly interleaved 8-byte values
    #
    if ndim == 3 and shape[2] in (3,4):
        x.shape = (shape[0], shape[1], shape[2] * 8)
    else:
        x.shape = (np.prod(shape[:-1]), shape[-1], 8)
    del data
    #
    # Compute A+B-C
    #
    a = np.zeros(x.shape, "i2")
    a[1:, :, :] = x[:-1, :, :]
    b = np.zeros(x.shape, "i2")
    b[:, 1:, :] = x[:, :-1, :]
    c = np.zeros(x.shape, "i2")
    c[1:, 1:, :] = x[:-1, :-1, :]
    p = a+b-c
    pick_a = (np.abs(a-p) <= np.abs(b-p)) & (np.abs(a-p) <= np.abs(c-p))
    pick_b = (~ pick_a) & (np.abs(b-p) <= np.abs(c-p))
    x[pick_a] -= a[pick_a]
    x[pick_b] -= b[pick_b]
    x[~(pick_a | pick_b)] -= c[~(pick_a | pick_b)]
    header = (
        np.array("CPNDIMG\0", "S8").tostring() +
        np.array([1], ">u4").tostring() +
        np.array([ndim], ">u4").tostring() +
        np.array(shape, ">u4").tostring())
        
    return header + zlib.compress(x.tostring())

def decode_image(data):
    '''Decode an image as encoded in encodeImage
    
    data - a string that contains the downloaded data
    
    returns an n-dimensional array as indicated by the encoding.
    '''
    cookie = np.frombuffer(data[0:8], "S8")
    if cookie != "CPNDIMG\0":
        raise ValueError('Image header cookie was not "CPNDIMG\\0"')
    version = np.frombuffer(data[8:12], ">u4")[0]
    if version != 1:
        raise ValueError("Unsupported version: %d" % version)
    ndim = np.frombuffer(data[12:16], ">u4")[0]
    shape = np.frombuffer(data[16:(16 + ndim*4)],">u4")
    x = np.frombuffer(zlib.decompress(data[(16 + ndim*4):]), "u1")
    x = np.ascontiguousarray(x.copy())
    if ndim == 3 and shape[2] in (3,4):
        bytes_per_pixel = shape[2] * 8
        x.shape = (shape[0], shape[1], bytes_per_pixel)
        raster_count = shape[0]
    else:
        x.shape = (np.prod(shape[:-1]), shape[-1], 8)
        raster_count = shape[-2]
    paeth_decoder(x, raster_count)
    x = np.fromstring(x.tostring(), ">f8")
    x.shape = shape
    return x

def stop_subimager():
    '''Stop the subimager process by web command'''
    
    conn = connect()
    conn.request("GET", "/stop")
    conn.getresponse()
    stop_semaphore.release()

__all__ = (start_subimager, get_image, get_metadata, post_image, 
           stop_subimager, HTTPError)

if __name__ == "__main__":
    import wx
    import matplotlib
    matplotlib.use("WX")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg, FigureManager
    sys.path.append(__root_path)
    from external_dependencies import fetch_external_dependencies
    
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        fetch_external_dependencies(overwrite=True)
        start_subimager()
    app = wx.PySimpleApp()
    frame = wx.Frame(None,size=(1024,768))
    menu = wx.Menu()
    metadata_item_id = wx.NewId()
    menu.Append(wx.ID_OPEN, "Open")
    menu.Append(metadata_item_id, "Open metadata")
    menu.Append(wx.ID_SAVEAS, "Save as")
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
            url = "file:" + urllib.pathname2url(dialog.Path)
            image = get_image(url, allowopenfiles="yes")
            frame.image = image
            image = image.astype(float)
            image /= np.max(image)
            frame.metadata = get_metadata(url, allowopenfiles="yes")
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
        
    def on_save_as(event, frame=frame):
        dialog = wx.FileDialog(frame, style = wx.FD_SAVE)
        dialog.Title = "Save image as"
        dialog.Wildcard = "JPEG (*.jpg)|*.jpg|TIFF (*.tif)|*.tif|PNG (*.png)|*.png"
        if dialog.ShowModal() == wx.ID_OK:
            url = "file:" + urllib.pathname2url(dialog.Path)
            post_image(url, frame.image, frame.metadata)
        dialog.Destroy()
    frame.Bind(wx.EVT_MENU, on_save_as, id=wx.ID_SAVEAS)
        
    def on_exit(event, frame=frame):
        frame.Close()
    frame.Bind(wx.EVT_MENU, on_exit, id=wx.ID_EXIT)
    frame.Layout()
    frame.Show()
    app.MainLoop()
    if subprocess_thread != None:
        stop_subimager()
        subprocess_thread.join()

