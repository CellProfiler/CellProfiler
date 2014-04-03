/**
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2014 Broad Institute
 * All rights reserved.
 * 
 * Please see the AUTHORS file for credits.
 * 
 * Website: http://www.cellprofiler.org
 */

package org.cellprofiler.imageset;

import net.imglib2.meta.TypedAxis;


/**
 * @author Lee Kamentsky
 *
 * A PlaneStack of ImagePlaneDetails. This class is mostly
 * here to give a class marker that can be used by
 * the StackAdapter filter predicate's getInputClass method.
 */
public class ImagePlaneDetailsStack extends PlaneStack<ImagePlaneDetails> {
	public ImagePlaneDetailsStack(final TypedAxis... axes){
		super(axes);
	}
	public boolean containsKey(String key) {
		for (ImagePlaneDetails ipd:this) {
			if (ipd.containsKey(key)) return true;
		}
		return false;
	}
	public String get(String key) {
		for (ImagePlaneDetails ipd:this) {
			final String value = ipd.get(key);
			if (value != null) return value;
		}
		return null;
	}
	/**
	 * Make a one-frame stack
	 * 
	 * @return
	 */
	static public ImagePlaneDetailsStack makeMonochromeStack(ImagePlaneDetails plane) {
		ImagePlaneDetailsStack stack = new ImagePlaneDetailsStack(XYAxes);
		stack.add(plane, 0, 0);
		return stack;
	}
	/**
	 * Make a color stack with one initial plane
	 * 
	 * @param plane
	 * @return a XYC stack containing the plane
	 */
	static public ImagePlaneDetailsStack makeColorStack(ImagePlaneDetails plane) {
		ImagePlaneDetailsStack stack = new ImagePlaneDetailsStack(XYCAxes);
		stack.add(plane, 0, 0, 0);
		return stack;
	}
}
