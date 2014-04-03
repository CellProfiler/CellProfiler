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

import java.util.List;

import org.cellprofiler.imageset.ImagePlaneDetails;
import org.cellprofiler.imageset.ImagePlaneDetailsStack;
import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;

/**
 * @author Lee Kamentsky
 * 
 * The StackAdapter adapts a ImagePlaneDetails filter predicate to a PlaneStack
 * of ImagePlaneDetails by evaluating each plane against the filter predicate.
 * 
 * The filter passes if every plane passes the adapted filter.
 *
 */
public class StackAdapter<TOUT> implements
		FilterPredicate<ImagePlaneDetailsStack, TOUT> {
	private FilterPredicate<ImagePlaneDetails, TOUT> planeFilterPredicate;
	/**
	 * Make a stack adapter for an ImagePlaneDetails filter predicate.
	 * 
	 * Doing all the type parameterization in here makes Java happy.
	 * 
	 * @param <T> The output type of both the StackAdapter and the FilterPredicate
	 * @param p
	 * @return
	 */
	static public <T> StackAdapter<T> makeAdapter(FilterPredicate<ImagePlaneDetails, T> p) {
		Class<T> klass = p.getOutputClass();
		StackAdapter<T> adapter = new StackAdapter<T>(klass);
		adapter.planeFilterPredicate = p;
		return adapter;
	}
	/**
	 * Constructor - adapt the given filter predicate to a stack
	 *  
	 * @param planeFilterPredicate Filter predicate to evaluate against
	 *                             every plane in a stack.
	 */
	protected StackAdapter(Class<TOUT> klass) {
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getSymbol()
	 */
	public String getSymbol() {
		return planeFilterPredicate.getSymbol();
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#setSubpredicates(java.util.List)
	 */
	public void setSubpredicates(List<FilterPredicate<TOUT, ?>> subpredicates)
			throws BadFilterExpressionException {
		planeFilterPredicate.setSubpredicates(subpredicates);
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#setLiteral(java.lang.String)
	 */
	public void setLiteral(String literal) throws BadFilterExpressionException {
		planeFilterPredicate.setLiteral(literal);
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#eval(java.lang.Object)
	 */
	public boolean eval(ImagePlaneDetailsStack candidate) {
		// TODO - URLPredicates need not be evaluated against every
		//        image plane, only the unique URLs in the stack.
		for (ImagePlaneDetails ipd:candidate) {
			if (! planeFilterPredicate.eval(ipd)) return false;
		}
		return true;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getInputClass()
	 */
	public Class<ImagePlaneDetailsStack> getInputClass() {
		return ImagePlaneDetailsStack.class;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getOutputClass()
	 */
	public Class<TOUT> getOutputClass() {
		return planeFilterPredicate.getOutputClass();
	}
	
}
