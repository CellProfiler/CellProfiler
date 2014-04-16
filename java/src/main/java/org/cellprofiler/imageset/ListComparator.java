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

import java.util.Comparator;
import java.util.List;

/**
 * @author Lee Kamentsky
 *
 * Applies a list of Comparator<T> sequentially to the elements
 * of two lists of T
 */
public class ListComparator<T> implements Comparator<List<T>> {
	final private List<Comparator<T>> comparators;
	/**
	 * Initialize with a list of comparators
	 * @param comparators
	 */
	public ListComparator(List<Comparator<T>> comparators) {
		this.comparators=comparators;
	}
	/* (non-Javadoc)
	 * @see java.util.Comparator#compare(java.lang.Object, java.lang.Object)
	 */
	public int compare(List<T> o1, List<T> o2) {
		for (int i=0; i<comparators.size(); i++) {
			final T s1 = o1.get(i);
			if (s1 == null) return -1;
			final T s2 = o2.get(i);
			if (s2 == null) return 1;
			int result = comparators.get(i).compare(s1, s2);
			if (result != 0) return result;
		}
		return 0;
	}

}
