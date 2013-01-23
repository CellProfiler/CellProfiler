/**
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2013 Broad Institute
 * All rights reserved.
 * 
 * Please see the AUTHORS file for credits.
 * 
 * Website: http://www.cellprofiler.org
 */
package org.cellprofiler.imageset;

import ome.xml.model.Image;
import ome.xml.model.OME;
import ome.xml.model.Plane;

/**
 * @author Lee Kamentsky
 * 
 * An image plane is a 2D monochrome or color plane within an image file.
 * 
 * Planes have a series # and image index and these can be used to
 * reference the particulars for the plane such as Z and T within
 * the image file metadata.
 *
 */
public class ImagePlane {
	private final ImageFile imageFile;
	private final int series;
	private final int index;
	/**
	 * Construct an image plane from a file, series and index
	 * 
	 * @param imageFile
	 * @param series
	 * @param index
	 */
	public ImagePlane(ImageFile imageFile, int series, int index) {
		this.imageFile = imageFile;
		this.series = series;
		this.index = index;
	}
	
	/**
	 * Construct the default image plane for a file
	 * 
	 * @param imageFile
	 */
	public ImagePlane(ImageFile imageFile) {
		this.imageFile = imageFile;
		this.series = 0;
		this.index = 0;
	}
	
	/**
	 * @return the image file containing this plane
	 */
	public ImageFile getImageFile() { return imageFile; }
	
	/**
	 * @return the plane's series
	 */
	public int getSeries() { return series; }
	
	/**
	 * @return the plane's index
	 */
	public int getIndex() { return index; }
	
	/**
	 * @return the OME model Image element that contains this plane 
	 */
	public Image getOMEImage() {
		final OME metadata = imageFile.getMetadata(); 
		if (metadata == null) return null;
		return metadata.getImage(series);
	}
	
	/**
	 * @return this plane's Plane element in the OME XML model
	 */
	public Plane getOMEPlane() {
		final Image image = getOMEImage();
		if (image == null) return null;
		return image.getPixels().getPlane(index);
	}
	@Override
	public String toString() {
		return String.format("ImagePlane: %s, series=%d, index=%d", imageFile.getURL(), series, index);
	}
}
