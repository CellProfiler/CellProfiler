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
 * This class represents the metadata for an image file.
 */
public class ImageFileDetails extends Details {
	final private ImageFile imageFile;
	public ImageFileDetails(ImageFile imageFile) {
		this.imageFile = imageFile;
	}
	
	public ImageFile getImageFile() {
		return imageFile;
	}
}
