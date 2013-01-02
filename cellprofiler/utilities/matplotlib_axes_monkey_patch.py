# Patches to prior versions of matplotlib
#
# Code below is adapted from matplotlib:
#
# This LICENSE AGREEMENT is between John D. Hunter (“JDH”), and the Individual
# or Organization (“Licensee”) accessing and otherwise using matplotlib software
# in source or binary form and its associated documentation.
#
# Subject to the terms and conditions of this License Agreement, 
# JDH hereby grants Licensee a nonexclusive, royalty-free, world-wide license to
# reproduce, analyze, test, perform and/or display publicly, prepare derivative 
# works, distribute, and otherwise use matplotlib 1.2.0 alone or in any 
# derivative version, provided, however, that JDH’s License Agreement and 
# JDH’s notice of copyright, i.e., “Copyright (c) 2002-2009 John D. Hunter; All 
# Rights Reserved” are retained in matplotlib 1.2.0 alone or in any derivative 
# version prepared by Licensee.
#
# In the event Licensee prepares a derivative work that is based on or 
# incorporates matplotlib 1.2.0 or any part thereof, and wants to make the 
# derivative work available to others as provided herein, then Licensee hereby 
# agrees to include in any such work a brief summary of the changes made to 
# matplotlib 1.2.0.
#
# JDH is making matplotlib 1.2.0 available to Licensee on an “AS IS” basis. 
# JDH MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR IMPLIED. BY WAY OF 
# EXAMPLE, BUT NOT LIMITATION, JDH MAKES NO AND DISCLAIMS ANY REPRESENTATION 
# OR WARRANTY OF MERCHANTABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT 
# THE USE OF MATPLOTLIB 1.2.0 WILL NOT INFRINGE ANY THIRD PARTY RIGHTS.
#
# JDH SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF MATPLOTLIB 1.2.0 
# FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS A RESULT OF 
# MODIFYING, DISTRIBUTING, OR OTHERWISE USING MATPLOTLIB 1.2.0, OR ANY 
# DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.
#
# This License Agreement will automatically terminate upon a material breach 
# of its terms and conditions.
#
# Nothing in this License Agreement shall be deemed to create any relationship 
# of agency, partnership, or joint venture between JDH and Licensee. This 
# License Agreement does not grant permission to use JDH trademarks or trade 
# name in a trademark sense to endorse or promote products or services of 
# Licensee, or any third party.
#
# By copying, installing or otherwise using matplotlib 1.2.0, Licensee agrees 
# to be bound by the terms and conditions of this License Agreement.

import math
import warnings

def set_adjustable(self, adjustable):
    """
    ACCEPTS: [ 'box' | 'datalim' | 'box-forced']
    """
    if adjustable in ('box', 'datalim', 'box-forced'):
        if self in self._shared_x_axes or self in self._shared_y_axes:
            if adjustable == 'box':
                raise ValueError(
                    'adjustable must be "datalim" for shared axes')
        self._adjustable = adjustable
    else:
        raise ValueError('argument must be "box", or "datalim"')

def apply_aspect(self, position=None):
    '''
    Use :meth:`_aspect` and :meth:`_adjustable` to modify the
    axes box or the view limits.
    '''
    if position is None:
        position = self.get_position(original=True)


    aspect = self.get_aspect()

    xscale, yscale = self.get_xscale(), self.get_yscale()
    if xscale == "linear" and yscale == "linear":
        aspect_scale_mode = "linear"
    elif xscale == "log" and yscale == "log":
        aspect_scale_mode = "log"
    elif (xscale == "linear" and yscale == "log") or \
             (xscale == "log" and yscale == "linear"):
        if aspect is not "auto":
            warnings.warn(
                'aspect is not supported for Axes with xscale=%s, yscale=%s' \
                % (xscale, yscale))
            aspect = "auto"
    else: # some custom projections have their own scales.
        pass

    if aspect == 'auto':
        self.set_position( position , which='active')
        return

    if aspect == 'equal':
        A = 1
    else:
        A = aspect

    #Ensure at drawing time that any Axes involved in axis-sharing
    # does not have its position changed.
    if self in self._shared_x_axes or self in self._shared_y_axes:
        if self._adjustable == 'box':
            self._adjustable = 'datalim'
            warnings.warn(
                'shared axes: "adjustable" is being changed to "datalim"')

    figW,figH = self.get_figure().get_size_inches()
    fig_aspect = figH/figW
    if self._adjustable in ['box', 'box-forced']:
        if aspect_scale_mode == "log":
            box_aspect = A * self.get_data_ratio_log()
        else:
            box_aspect = A * self.get_data_ratio()
        pb = position.frozen()
        pb1 = pb.shrunk_to_aspect(box_aspect, pb, fig_aspect)
        self.set_position(pb1.anchored(self.get_anchor(), pb), 'active')
        return

    # reset active to original in case it had been changed
    # by prior use of 'box'
    self.set_position(position, which='active')


    xmin,xmax = self.get_xbound()
    ymin,ymax = self.get_ybound()

    if aspect_scale_mode == "log":
        xmin, xmax = math.log10(xmin), math.log10(xmax)
        ymin, ymax = math.log10(ymin), math.log10(ymax)

    xsize = max(math.fabs(xmax-xmin), 1e-30)
    ysize = max(math.fabs(ymax-ymin), 1e-30)


    l,b,w,h = position.bounds
    box_aspect = fig_aspect * (h/w)
    data_ratio = box_aspect / A

    y_expander = (data_ratio*xsize/ysize - 1.0)
    # If y_expander > 0, the dy/dx viewLim ratio needs to increase
    if abs(y_expander) < 0.005:
        return

    if aspect_scale_mode == "log":
        dL = self.dataLim
        dL_width = math.log10(dL.x1) - math.log10(dL.x0)
        dL_height = math.log10(dL.y1) - math.log10(dL.y0)
        xr = 1.05 * dL_width
        yr = 1.05 * dL_height
    else:
        dL = self.dataLim
        xr = 1.05 * dL.width
        yr = 1.05 * dL.height

    xmarg = xsize - xr
    ymarg = ysize - yr
    Ysize = data_ratio * xsize
    Xsize = ysize / data_ratio
    Xmarg = Xsize - xr
    Ymarg = Ysize - yr
    xm = 0  # Setting these targets to, e.g., 0.05*xr does not seem to help.
    ym = 0

    changex = (self in self._shared_y_axes
               and self not in self._shared_x_axes)
    changey = (self in self._shared_x_axes
               and self not in self._shared_y_axes)
    if changex and changey:
        warnings.warn("adjustable='datalim' cannot work with shared "
                      "x and y axes")
        return
    if changex:
        adjust_y = False
    else:
        if xmarg > xm and ymarg > ym:
            adjy = ((Ymarg > 0 and y_expander < 0)
                    or (Xmarg < 0 and y_expander > 0))
        else:
            adjy = y_expander > 0
        adjust_y = changey or adjy  #(Ymarg > xmarg)
    if adjust_y:
        yc = 0.5*(ymin+ymax)
        y0 = yc - Ysize/2.0
        y1 = yc + Ysize/2.0
        if aspect_scale_mode == "log":
            self.set_ybound((10.**y0, 10.**y1))
        else:
            self.set_ybound((y0, y1))
    else:
        xc = 0.5*(xmin+xmax)
        x0 = xc - Xsize/2.0
        x1 = xc + Xsize/2.0
        if aspect_scale_mode == "log":
            self.set_xbound((10.**x0, 10.**x1))
        else:
            self.set_xbound((x0, x1))


'''
Monkey patch these methods in matplotlib.Axes with the same methods from
matplotlib revision 8079.
'''
import matplotlib
if matplotlib.__version__ < "0.99.1.3.rc1":
    import matplotlib.axes
    matplotlib.axes.Axes.set_adjustable = set_adjustable
    matplotlib.axes.Axes.apply_aspect = apply_aspect
