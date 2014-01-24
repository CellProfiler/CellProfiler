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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.cellprofiler.imageset.ImagePlane;

/**
 * @author Lee Kamentsky
 *
 * IndexedImagePlaneDetails adds an index to ImagePlaneDetails
 * for use by CellProfiler. The index lets CellProfiler retrieve
 * the Python object, given the index.
 */
public class IndexedImagePlaneDetails extends ImagePlaneDetails {
	final int index;
	/**
	 * Constructor taking an index in addition to the image plane and metadata
	 * @param imagePlane
	 * @param metadata
	 * @param index
	 */
	public IndexedImagePlaneDetails(ImagePlane imagePlane,
			Map<String, String> metadata, int index) {
		super(imagePlane, metadata);
		this.index = index;
	}
	
	public IndexedImagePlaneDetails(ImagePlaneDetails ipd, int index) {
		super(ipd.imagePlane, ipd.metadata);
		this.index = index;
	}
	
	/**
	 * @return the index passed in by the constructor
	 */
	public int getIndex() {
		return index;
	}
	
	/**
	 * Convert a list of ImagePlaneDetails into a list of IndexedImagePlaneDetails
	 * where the index is the object's order in the list.
	 * 
	 * @param ipds
	 * @return copy of the list passed in, replacing ImagePlaneDetails with IndexedImagePlaneDetails
	 */
	public static List<IndexedImagePlaneDetails> index(List<ImagePlaneDetails> ipds) {
		List<IndexedImagePlaneDetails> result = new ArrayList<IndexedImagePlaneDetails>(ipds.size());
		int index = 0;
		for (ImagePlaneDetails ipd:ipds) {
			result.add(new IndexedImagePlaneDetails(ipd, index++));
		}
		return result;
	}

	/**
	 * Retrieve the indexes from a list of ImagePlaneDetails.
	 * @param ipds a list of indexed image plane details
	 * @param indices an array of the size of the list. The array is filled with
	 *        indices stored in the indexed IPD or is -1 if the entry is null.
	 * 
	 * Fill the indices in the array with the indices in the IPDs on the list
	 */
	public static void getIndices(List<IndexedImagePlaneDetails> ipds, int [] indices) {
		for (int i=0; i<ipds.size(); i++) {
			final IndexedImagePlaneDetails ipd = ipds.get(i);
			indices[i] = (ipd == null)? -1:ipd.index;
		}
	}
}
