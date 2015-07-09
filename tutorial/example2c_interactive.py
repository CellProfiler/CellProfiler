'''<b>Example2c</b> User interaction with an image.

Sometimes, it's useful to have a semi-automated or completely manual step
in your pipeline. CellProfiler <i>can</i> do this, but is designed to
run the analysis hands-off, so you need to handle user interaction in a
controlled manner. The upcoming multiprocessing changes make this necessary,
but also have support for this.

There are two steps to writing a successful interaction. The first is to
call workspace.interaction_request inside the module's run method. The second
is to write your interaction_handler to handle the interaction reequest.

You can pass numpy arrays, strings, numbers or lists, dictionaries or tuples
of the above as arguments and they will get pickled and unpickled correctly.
The interaction_handler can return the same.

In this example, the module's "run" method passes the input image to the
interaction handler. The interaction handler shows a dialog box with a
user interface that lets the user draw rectangular regions. The interaction
handler fills these regions with the value, "1". The interaction handler
returns the edited image to the "run" method after the user closes the
dialog.

<b>Note</b> This will only work with the new multiprocessing code.
'''

import numpy as np
# We get the CPModule class from here. "cpm" is the standard alias for the
# Python module, "cellprofiler.cpmodule".

import cellprofiler.cpmodule as cpm

# This is where all settings are defined. See below for explanation.

import cellprofiler.settings as cps

import cellprofiler.cpimage as cpi

#
# This is the module class definition. Each module is a Python class
# whose base class is CPModule. All that means is that anything that
# CellProfiler expects to find in a module will use whatever is in
# CPModule unless you define an override of it in your module
#
class Example2c(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Example2c"
    category = "Image Processing"

    def create_settings(self):
        self.input_image_name = cps.ImageNameSubscriber("Input image", "DNA")
        self.output_image_name = cps.ImageNameProvider("Output image", "Output")
        
    def settings(self):
        return [self.input_image_name, self.output_image_name]
    
    def run(self, workspace):
        #
        # Get the image pixels from the image set
        #
        image_set = workspace.image_set
        image = image_set.get_image(self.input_image_name.value)
        pixel_data = image.pixel_data
        #
        # Call the interaction handler with the image. The interaction
        # handler will be invoked - that might be in a separate thread,
        # a separate process or on an entirely different computer.
        #
        result = workspace.interaction_request(self, pixel_data)
        if workspace.show_frame:
            workspace.display_data.output = result
        output_image = cpi.Image(result)
        image_set.add(self.output_image_name.value, output_image)
        
    def handle_interaction(self, pixel_data):
        #
        # This gets called in the UI thread and we're allowed to import
        # UI modules such as WX or Matplotlib and pop up windows.
        #
        # The documentation for the Python WX widgets is hosted at:
        #
        # http://www.wxpython.org/docs/api/wx-module.html
        #
        # The documentation for Matplotlib is hosted at:
        #
        # http://matplotlib.org/api
        #
        # The Matplotlib examples are often useful because they show the
        # "happy path" - the well-trodden way that people have done things
        # is generally the best choice because it demonstrably works.
        #
        # http://matplotlib.org/examples/index.html
        #
        import wx
        import matplotlib
        import matplotlib.lines
        import matplotlib.cm
        import matplotlib.backends.backend_wxagg
        #
        # Make a wx.Dialog. "with" will garbage collect all of the
        # UI resources when the user closes the dialog.
        #
        # This is how our dialog is structured:
        #
        # -------- WX Dialog frame ---------
        # |                                |
        # |  ----- WX BoxSizer ----------  |
        # |  |                          |  |
        # |  |  -- Matplotlib canvas -- |  |
        # |  |  |                     | |  |
        # |  |  |  ---- Figure ------ | |  |
        # |  |  |  |                | | |  |
        # |  |  |  |  --- Axes ---- | | |  |
        # |  |  |  |  |           | | | |  |
        # |  |  |  |  | AxesImage | | | |  |
        # |  |  |  |  |           | | | |  |
        # |  |  |  |  | Line2D    | | | |  |
        # |  |  |  |  ------------- | | |  |
        # |  |  |  |                | | |  |
        # |  |  |  ------------------ | |  |
        # |  |  ----------------------- |  |
        # |  |                          |  |
        # |  |  | WX StdDlgButtonSizer| |  |
        # |  |  |                     | |  |
        # |  |  | -- WX OK Button --- | |  |
        # |  |  | |                 | | |  |
        # |  |  | ------------------- | |  |
        # |  |  ----------------------- |  |
        # |  ----------------------------  |
        # ----------------------------------
        #
        with wx.Dialog(None, 
                       title="Edit image", 
                       size=(640, 720)) as dlg:
            #
            # A wx.Sizer lets you automatically adjust the size
            # of a window's subwindows.
            #
            dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
            #
            # We draw on the figure
            #
            figure = matplotlib.figure.Figure()
            #
            # Define an Axes on the figure
            #
            axes = figure.add_axes((.05, .05, .9, .9))
            axes.imshow(pixel_data, cmap=matplotlib.cm.gray)
            #
            # The canvas renders the figure
            #
            canvas = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(
                dlg, -1, figure)
            #
            # Put the canvas in the dialog
            #
            dlg.Sizer.Add(canvas, 1, wx.EXPAND)
            #
            # This is a button sizer and it handles the OK button at the bottom.
            # WX will fill in the appropriate names for buttons in this sizer:
            # wx.ID_OK = OK button
            # wx.ID_CANCEL = Cancel button
            # wx.ID_YES / wx.ID_NO
            #
            button_sizer = wx.StdDialogButtonSizer()
            dlg.Sizer.Add(button_sizer, 0, wx.EXPAND)
            ok_button = wx.Button(dlg, wx.ID_OK)
            button_sizer.AddButton(ok_button)
            button_sizer.Realize()
            #
            # "on_button" gets called when the button is pressed.
            #
            # ok_button.Bind directs WX to handle a button press event
            # by calling "on_button" with the event.
            #
            # dlg.EndModal tells WX to close the dialog and return control
            # to the caller.
            #
            def on_button(event):
                dlg.EndModal(1)
            ok_button.Bind(wx.EVT_BUTTON, on_button)
            #
            # This is a rudimentary Matplotlib event handler that:
            # * initiates rectangle drawing when you press the mouse button
            # * presents feedback as you drag with the mouse
            # * draws the rectangle when you release the mouse button.
            #
            # Also look at: 
            # http://matplotlib.org/examples/event_handling/lasso_demo.html
            #
            # for an alternative to this.
            #
            # We keep the state of the mouse in a dictionary
            #
            # d["start"] has the coordinates of the start pixel of the rectangle
            #
            d = dict(start=None, end=None, rectangle=None)
            #
            # When the user presses a mouse button, record the start coordinate
            #
            def on_mouse_down(event):
                #
                # Only do this if the mouse is positioned over the image
                #
                if event.inaxes == axes:
                    d["start"] = (event.ydata, event.xdata)
                    #
                    # Also capture the mouse. All mouse events go to
                    # the canvas window until canvas.ReleaseMouse is
                    # called.
                    #
                    canvas.CaptureMouse()
             
            #
            # Keep track of mouse movement when drawing using a rectangle
            #
            def on_mouse_move(event):
                #
                # Only do this if the mouse is positioned over the image and
                # we have captured the mouse.
                #
                if event.inaxes == axes and d["start"] is not None:
                    #
                    # Store the coordinates of the current position here
                    #
                    d["end"] = (event.ydata, event.xdata)
                    #
                    # Make a 2x2 Numpy array of the mouse coordinates
                    #
                    coords = np.array([d["start"], d["end"]])
                    #
                    # Pick out the X coordinates of the rectangle. We have 5
                    # points with the first and last being the same to close
                    # the rectangle:
                    #
                    #  2 ------- 3
                    #  |         |
                    #  |         |
                    #  |         |
                    #  1,5 ----- 4
                    #
                    x = [coords[0, 1], coords[0, 1],
                         coords[1, 1], coords[1, 1],
                         coords[0, 1]]
                    y = [coords[0, 0], coords[1, 0],
                         coords[1, 0], coords[0, 0],
                         coords[0, 0]]
                    if d["rectangle"] is None:
                        #
                        # Create the rectangle if there is none
                        # 
                        # Line2D is an "artist" composed of connected line
                        # segments. Artists are axes annotations that provide
                        # the visualization that appears inside the axes.
                        #
                        # See http://matplotlib.org/api/artist_api.html
                        #
                        # for links to the different artists you can use.
                        #
                        d["rectangle"] = \
                            matplotlib.lines.Line2D(x, y, color="red")
                        #
                        # Add the artist to the axes
                        #
                        axes.add_line(d["rectangle"])
                    else:
                        #
                        # You can update the rectangle appearance by rewriting
                        # the line data.
                        #
                        line = d["rectangle"]
                        line.set_xdata(x)
                        line.set_ydata(y)
                    #
                    # Redraw the figure after changing it. Matplotlib draws
                    # both the AxesImage added using imshow and the rectangle.
                    #
                    canvas.draw()
                    #
                    # "Refresh" tells WX to redisplay the canvas.
                    #
                    canvas.Refresh()
            
            #
            # When the user lifts the mouse, record the rectangle
            #
            def on_mouse_up(event):
                if d["rectangle"] is not None:
                    #
                    # Find the rectangle's minimum and maximum extents
                    #
                    min_i = int(min(d["start"][0], d["end"][0]))
                    min_j = int(min(d["start"][1], d["end"][1]))
                    max_i = int(max(d["start"][0], d["end"][0]))+1
                    max_j = int(max(d["start"][1], d["end"][1]))+1
                    #
                    # Set the region between the minimum (inclusive) and the
                    # maximum (exclusive) to 1
                    #
                    pixel_data[min_i:max_i, min_j:max_j] = 1
                    #
                    # Tell the Line2D artist to remove itself from the axes.
                    #
                    d["rectangle"].remove()
                    #
                    # We set the "rectangle" and "start" slots of the dictionary
                    # to "None". This tells the event handlers that we're in
                    # the initial state once again - no rectangle on the
                    # screen, no mouse capture and no start chosen
                    #
                    d["rectangle"] = None
                    d["start"] = None
                    #
                    # We updated the image, so replace the old AxesImage with
                    # the new one. Draw and refresh.
                    #
                    axes.imshow(pixel_data, cmap=matplotlib.cm.gray)
                    canvas.draw()
                    canvas.Refresh()
                #
                # Remember to give the mouse back to the operating system.
                #
                if canvas.HasCapture():
                    canvas.ReleaseMouse()
            #
            # Matplotlib uses mpl_connect to connect window events to
            # event handling functions.
            #
            canvas.mpl_connect('button_press_event', on_mouse_down)
            canvas.mpl_connect('button_release_event', on_mouse_up)
            canvas.mpl_connect('motion_notify_event', on_mouse_move)
            #
            # Layout and show the dialog. The WX Layout() method tells
            # WX to use the sizers to place the dialog's controls:
            # the canvas and OK button.
            #
            dlg.Layout()
            dlg.ShowModal()
            #
            # Return the image
            #
            return pixel_data
        
    def display(self, workspace, figure):
        figure.set_subplots((1,1))
        figure.subplot_imshow_grayscale(0, 0, workspace.display_data.output)
                    