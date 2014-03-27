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

package org.cellprofiler.imageset.filter;

import net.imglib2.Axis;

import org.cellprofiler.imageset.PlaneStack;

/**
 * @author Lee Kamentsky
 *
 * A PlaneStack of ImagePlaneDetails. This class is mostly
 * here to give a class marker that can be used by
 * the StackAdapter filter predicate's getInputClass method.
 */
public class ImagePlaneDetailsStack extends PlaneStack<ImagePlaneDetails> {
	public ImagePlaneDetailsStack(final Axis... axes){
		super(axes);
	}
}
