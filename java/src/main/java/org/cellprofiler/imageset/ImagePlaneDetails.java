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


/**
 * @author Lee Kamentsky
 *
 * The ImagePlaneDetails class couples an ImagePlane to the
 * metadata extracted from the ImagePlane by the extractors.
 */
public class ImagePlaneDetails extends Details {
	/**
	 * The image plane to be filtered 
	 */
	private final ImagePlane imagePlane;
	public ImagePlaneDetails(ImagePlane imagePlane, Details parent) {
		super(parent);
		this.imagePlane = imagePlane;
	}
	protected ImagePlaneDetails(ImagePlane imagePlane, Details parent, Details child) {
		super(parent, child);
		this.imagePlane = imagePlane;
	}
	public ImagePlane getImagePlane() {
		return imagePlane;
	}
	
	/**
	 * Sometimes it is necessary to force a color interpretation
	 * on an image plane details by cloning it and replacing
	 * the image plane with one of a different interpretation
	 * 
	 * (insert emoji of cat throwing up please)
	 * 
	 * @return either self if already monochrome or a clone
	 */
	public ImagePlaneDetails coerceToMonochrome() {
		if (imagePlane.getChannel() == ImagePlane.INTERLEAVED) {
			final ImagePlane planeCopy = new ImagePlane(
					imagePlane.getSeries(), imagePlane.getIndex(), ImagePlane.ALWAYS_MONOCHROME);
			return new ImagePlaneDetails(planeCopy, parent, this);
		} else {
			return this;
		}
	}
	/**
	 * Sometimes it is necessary to force a color interpretation
	 * on an image plane details by cloning it and replacing
	 * the image plane with one of a different interpretation
	 * 
	 * (insert emoji of cat throwing up please)
	 * 
	 * @return either self if already monochrome or a clone
	 */
	public ImagePlaneDetails coerceToColor() {
		if (imagePlane.getChannel() != ImagePlane.INTERLEAVED) {
			final ImagePlane planeCopy = new ImagePlane(
					imagePlane.getSeries(), imagePlane.getIndex(), ImagePlane.INTERLEAVED);
			return new ImagePlaneDetails(planeCopy, parent, this);
		} else {
			return this;
		}
	}
}
