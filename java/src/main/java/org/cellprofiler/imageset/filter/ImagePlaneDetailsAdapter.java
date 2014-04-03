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


import org.cellprofiler.imageset.ImageFile;
import org.cellprofiler.imageset.ImagePlaneDetails;

/**
 * @author Lee Kamentsky
 *
 * The ImagePlaneDetailsAdapter adapts an ImageFile filter predicate
 * to an ImagePlaneDetails's ImageFile.
 * 
 */
public class ImagePlaneDetailsAdapter<TOUT> 
	extends FilterAdapter<ImagePlaneDetails, ImageFile, TOUT> {
	/**
	 * Make an adapter for an ImageFile filter predicate.
	 * 
	 * Doing all the type parameterization in here makes Java happy.
	 * 
	 * @param <T> The output type of both the StackAdapter and the FilterPredicate
	 * @param p
	 * @return
	 */
	static public <T> ImagePlaneDetailsAdapter<T> makeAdapter(FilterPredicate<ImageFile, T> p) {
		Class<T> klass = p.getOutputClass();
		ImagePlaneDetailsAdapter<T> adapter = new ImagePlaneDetailsAdapter<T>(klass);
		adapter.predicate = p;
		return adapter;
	}
	/**
	 * Initialize the adapter, using the output class to get the
	 * correct type parameterization
	 * 
	 * @param klass
	 */
	protected ImagePlaneDetailsAdapter(Class<TOUT> klass) {
		super(ImagePlaneDetails.class, ImageFile.class, klass);
	}
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getInputClass()
	 */
	public Class<ImagePlaneDetails> getInputClass() {
		return ImagePlaneDetails.class;
	}
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterAdapter#getValue(java.lang.Object)
	 */
	@Override
	public ImageFile getValue(ImagePlaneDetails candidate) {
		return candidate.getImagePlane().getImageFile();
	}

}
