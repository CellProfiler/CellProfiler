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

import java.util.List;

import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;

abstract public class AbstractURLPredicateBase  implements
	FilterPredicate<ImagePlaneDetails, String> {

	protected FilterPredicate<String, ?> subpredicate;

	public AbstractURLPredicateBase() {
		super();
	}

	public void setSubpredicates(List<FilterPredicate<String, ?>> subpredicates)
			throws BadFilterExpressionException {
				if (subpredicates.size() != 1) {
					throw new BadFilterExpressionException(String.format("The %s predicate takes a single subpredicate", getSymbol()));
				}
				subpredicate = subpredicates.get(0);
			}

	public void setLiteral(String literal) throws BadFilterExpressionException {
		throw new BadFilterExpressionException(String.format("The %s predicate does not take a literal", getSymbol()));
	}

	public Class<ImagePlaneDetails> getInputClass() {
		return ImagePlaneDetails.class;
	}

	public Class<String> getOutputClass() {
		return String.class;
	}

}