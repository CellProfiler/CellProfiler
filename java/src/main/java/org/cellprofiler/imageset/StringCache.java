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

import java.lang.ref.WeakReference;
import java.util.WeakHashMap;

/**
 * @author Lee Kamentsky
 *
 * Maintain a weak-referenced collection of "interned"
 * strings to conserve memory.
 */
public class StringCache {
	private static final WeakHashMap<String, WeakReference<String>> cache = 
		new WeakHashMap<String, WeakReference<String>>();
	
	/**
	 * Find or create the single reference to the sequence in the cache
	 * 
	 * @param s the string to be interned
	 * @return the canonical object for that string
	 */
	public static final String intern(CharSequence s) {
		if (cache.containsKey(s)) {
			return cache.get(s).get();
		}
		final String sString = s.toString();
		cache.put(sString, new WeakReference<String>(sString));
		return sString;
	}

}
