/**
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2015 Broad Institute
 * All rights reserved.
 * 
 * Please see the AUTHORS file for credits.
 * 
 * Website: http://www.cellprofiler.org
 */

package org.cellprofiler.imageset;


/**
 * @author Lee Kamentsky
 *
 * The ImageSeries details holds metadata for
 * one of the series (or the Image in OME-speak) 
 * within an image file.
 */
public class ImageSeriesDetails extends Details {
	final private ImageSeries imageSeries;
	public ImageSeriesDetails(ImageSeries imageSeries, Details parent) {
		super(parent);
		this.imageSeries = imageSeries;
	}
	
	public ImageSeries getImageSeries() {
		return imageSeries;
	}
}
