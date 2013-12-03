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
package org.cellprofiler.imageset.filter;

import org.cellprofiler.imageset.OMEMetadataExtractor;

/**
 * @author Lee Kamentsky
 * 
 * A predicate that determines whether an image "plane"
 * is really a stack of them.
 * 
 * An image "plane" is a stack if it has more than one
 * Z or T slice as indicated by the values of the
 * MD_SIZE_Z or MD_SIZE_T metadata.
 *
 */
public class IsStackPredicate extends
		AbstractTerminalPredicate<ImagePlaneDetails> {
	final static public String SYMBOL="isstack";

	protected IsStackPredicate() {
		super(ImagePlaneDetails.class);
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getSymbol()
	 */
	public String getSymbol() {
		return SYMBOL;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#eval(java.lang.Object)
	 */
	public boolean eval(ImagePlaneDetails candidate) {
		for (final String key: new String [] {OMEMetadataExtractor.MD_SIZE_T, OMEMetadataExtractor.MD_SIZE_Z} ) {
			if (! candidate.metadata.containsKey(key)) continue;
			final String value = candidate.metadata.get(key);
			try {
				if (Integer.parseInt(value) > 1) return true;
			} catch (NumberFormatException e) {
				continue;
			}
		}
		return false;
	}
}
