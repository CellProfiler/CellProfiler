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
package org.cellprofiler.imageset.filter;


import org.cellprofiler.imageset.ImageFile;

/**
 * @author Lee Kamentsky
 *
 */
public abstract class AbstractURLPredicate extends AbstractURLPredicateBase {
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#eval(java.lang.Object)
	 */
	public boolean eval(ImageFile candidate) {
		String value = getValue(candidate);
		if (value == null) return false;
		return subpredicate.eval(value);
	}

	protected abstract String getValue(ImageFile candidate);

}
