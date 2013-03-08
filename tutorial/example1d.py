'''<b>Example1d</b> CellProfiler display
<hr>
This module demonstrates some things you can do with CellProfiler
and its displays.
'''

import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

D_COLOR_IMAGE = "Color image"
D_GRAYSCALE_IMAGE = "Grayscale image"
D_TYPES_OF_PLOTS = "Plots"

class Example1d(cpm.CPModule):
    variable_revision_number = 1
    module_name = "Example1d"
    category = "Other"
    
    def create_settings(self):
        self.display_choice = cps.Choice("Display type",
            [D_COLOR_IMAGE, D_GRAYSCALE_IMAGE, D_TYPES_OF_PLOTS])
        
    def settings(self):
        return [self.display_choice]
    
    def run(self, workspace):
        #
        # The key to successful display in CellProfiler is to save all of
        # the state that's needed in workspace.display_data. You can save
        # numbers, strings, lists, tuples, sets, dictionaries and
        # numpy arrays.
        #
        # It might not be apparent why we require two separate routines, "run"
        # and "display". In test mode, "display" is run in the same thread,
        # immediately after "run". However, in the first released versions,
        # "run" is run in a background thread in analysis mode and "display"
        # is run in the UI thread. This keeps the UI active during analysis,
        # but it means that "display" could happen at any time after "run",
        # including during the next cycle. That means that you have to
        # put the data needed for display some place that won't change.
        # The situation is even more severe in the multiprocessing scenario
        # where "run" is executed in a worker process and "display" is executed
        # in the user interface process.
        #
        #
        # Put different things in the workspace depending on the display choice.
        # 
        if self.display_choice == D_GRAYSCALE_IMAGE:
            #
            # Make up some fake data - in this case, a circle.
            #
            # Numpy's ndarray is well-suited to image processing - in
            # CellProfiler, an image is a 2-d array of grayscale values.
            # Here, we use something called "mgrid" to make two arrays,
            # one which has the "i" coordinate of the array index and
            # one which has the "j" coordinate. Numpy uses "i" and "j" = 
            # row and column as opposed to Matplotlib which uses X and Y.
            #
            i, j = np.mgrid[-250:251, -250:251]
            #
            # Now let's make an array whose value is the inverse of the
            # distance from the center + a constant to avoid divide-by zero
            #
            pixel_data = 1 / (1+ np.sqrt(i*i + j*j))
            #
            # Let's restrict the circle to a radius of 50. The following says
            # For pixel_data elements whose distance is > 50, make them zero.
            #
            pixel_data[np.sqrt(i*i + j*j) > 50] = 0
            #
            # Store the numpy array in the workspace
            #
            workspace.display_data.grayscale = pixel_data
            #
            # Save the layout to use for this display
            #
            workspace.display_data.subplots = (1, 1)
        elif self.display_choice == D_COLOR_IMAGE:
            #
            # Let's make red, green and blue circles
            #
            # There are 3 dimensions here: 300 x 500 x 3
            #
            i, j, k = np.mgrid[0:300, 0:500, 0:3]
            #
            # Let's pick 3 random centers + a radius for each
            #
            # 3 values between 0 and the length of i which is 300
            i_center = np.random.randint(0, len(i), len(k)) 
            # 3 values between 0 and the length of j which is 500
            j_center = np.random.randint(0, len(j), len(k))
            # 3 values between 25 and 50
            #
            radius = np.random.randint(25, 50, len(k))
            #
            # OK - K picks the circle for the color and the mask is true
            #      whenever a point is less than the radius from its center.
            #      All done at one go for 3 circles, neat, eh?
            #
            mask = (i - i_center[k])** 2 + (j - j_center[k])** 2 < radius[k]**2
            #
            # Make some random data and mask it to get circles
            #
            pixel_data = np.random.uniform(size = i.shape)
            # The following means
            # for every element of pixel data for which the mask is false,
            # make it zero
            pixel_data[~mask] = 0
            workspace.display_data.color = pixel_data
            workspace.display_data.subplots = (2, 2)
        elif self.display_choice == D_TYPES_OF_PLOTS:
            workspace.display_data.A = np.random.uniform(
                low = -2, high = 2, size=100)
            workspace.display_data.B = np.random.normal(size=100)
            workspace.display_data.subplots = (4, 1)
    
    #
    # The signature for display will change in the next release of CellProfiler
    # Uncomment the following if you're using that version
    #
    def display(self, workspace, figure):
        #
        # A subplot is a matplotlib Axes. set_subplots sets up the arrangement
        # of them. The first number in the tuple is the number of columns
        # and the second is the number of rows.
        #
        figure.set_subplots(workspace.display_data.subplots)
    #
    # This is the signature for the current CP release
    #
    #def display(self, workspace):
        ##
        ## create_or_find_figure will either create a new figure and window
        ## for the module or reuse a previously-created one. See above
        ## for discussion of subplots
        ##
        #figure = workspace.create_or_find_figure(
            #subplots = workspace.display_data.subplots)
        if self.display_choice == D_GRAYSCALE_IMAGE:
            figure.subplot_imshow_grayscale(
                0, 0,                             # the subplot coordinates
                workspace.display_data.grayscale, # the image
                title = "Grayscale")              # the subplot title
        elif self.display_choice == D_COLOR_IMAGE:
            pixel_data = workspace.display_data.color
            ax11 = figure.subplot_imshow_color(1, 1, pixel_data,
                                               title = "Color")
            colors = ("Red", "Green", "Blue")
            for k in range(pixel_data.shape[2]):
                figure.subplot_imshow_grayscale(
                    int(k / 2), k % 2, 
                    pixel_data[:, :, k],
                    title = colors[k],
                    sharex = ax11, # share(x,y) tell Matplotlib to make all
                    sharey = ax11) # axes share the same extents so zooming
                                   # and panning of one will make them all
                                   # zoom and pan.
        elif self.display_choice == D_TYPES_OF_PLOTS:
            a = workspace.display_data.A
            b = workspace.display_data.B
            #
            # Matplotlib has lots of ways to plot data. See if you
            # can improve these.
            #
            # Scatterplot:
            #
            ax00 = figure.subplot(0, 0)
            ax00.plot(a, b, 
                      'go',              # means "green circles"
                      linestyle=" ") # means "invisible lines between points"
            ax00.set_xlabel("A")
            ax00.set_ylabel("B")
            #
            # Here's a box plot. The first argument can be either a single
            # vector of values to plot or a sequence of them, in which case
            # you get one boxplot for each
            #
            ax01 = figure.subplot(1, 0)
            ax01.boxplot((a, b))
            ax01.set_xticklabels(("A", "B"))
            #
            # Two overlayed histograms
            #
            ax10 = figure.subplot(2, 0)
            ax10.hist(a, label = "A")
            ax10.hist(b, label = "B")
            ax10.legend()
            #
            # A table - boy is it ugly, can you make it better?
            #
            figure.subplot_table(
                3, 0, 
                [("","A", "B"),
                 ("mean", str(a.mean()), str(b.mean())),
                 ("min", str(a.min()), str(b.min())),
                 ("max", str(a.max()), str(b.max())),
                 ("st dev", str(np.std(a)), str(np.std(b)))],
                ratio = (.2, .4, .4))
                                  
