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

import java.util.List;

import org.cellprofiler.imageset.filter.ImagePlaneDetails;

/**
 * @author Lee Kamentsky
 *
 * Report an error such as a duplicate or missing image in an image set
 */
public class ImageSetError {
	/**
	 * The key of metadata values that defines the image set,
	 * for instance { "Plate1", "A01" } for metadata keys, "Plate" and "Well"
	 */
	final private List<String> key;
	/**
	 * The name of the channel which has missing or duplicate entries.
	 */
	final private String channelName;
	/**
	 * The error message 
	 */
	final private String message;
	private List<ImagePlaneDetails> imageSet;
	public ImageSetError(String channelName, String message, List<String> key) {
		this.channelName = channelName;
		this.message = message;
		this.key = key;
	}
	public String getChannelName() { return channelName; }
	public String getMessage() { return message; }
	public List<String> getKey() { return key; }
	
	@Override
	public String toString() { return message; }
	/**
	 * Record the image planes that were not in error.
	 * 
	 * @param imageSet a list of image planes that were not in error for the key
	 *                 with the error channels represented as nulls.
	 */
	void setImageSet(List<ImagePlaneDetails> imageSet) {
		this.imageSet = imageSet;
	}
	/**
	 * @return a list of correctly-discovered image planes for the image set
	 */
	List<ImagePlaneDetails> getImageSet() {
		return imageSet;
	}
}
