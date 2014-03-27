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


/**
 * @author Lee Kamentsky
 * 
 * A predicate that determines whether a file name has an extension determined 
 * by the subexpression that follows.
 *
 */
public class ExtensionPredicate extends AbstractURLPredicateBase {
	final static public String SYMBOL = "extension";
	
	public String getSymbol() {
		return SYMBOL;
	}

	public boolean eval(ImageFile candidateFile) {
		String candidate = candidateFile.getFileName();
		if (candidate == null) return false;
		int index = candidate.length();
		while (index > 0) {
			index = candidate.lastIndexOf(".", index-1);
			if (index < 0) break;
			if (subpredicate.eval(candidate.substring(index+1))) return true;
		}
		return false;
	}
}
