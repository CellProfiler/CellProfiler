/**
 * 
 */
package org.cellprofiler.imageset.filter;

import java.util.List;

import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;

/**
 * The LogicPredicate performs a logical operation on its subpredicates
 * to arrive at the filter result.
 * 
 * @author Lee Kamentsky
 *
 * @param <TINOUT>
 */
public abstract class LogicPredicate<TINOUT> implements FilterPredicate<TINOUT, TINOUT> {
	final Class<TINOUT> klass;
	List<FilterPredicate<TINOUT, ?>> subpredicates;
	/**
	 * Constructor - choose the class for the input candidates of both the logic predicate
	 *               and subpredicates.
	 *               
	 * @param klass
	 */
	public LogicPredicate(Class<TINOUT> klass) {
		this.klass = klass;
	}

	public void setLiteral(String literal) throws BadFilterExpressionException {
		throw new AssertionError("Logic predicates are not qualified by literals");
	}

	/**
	 * The derived classes evaluate the boolean results of running the subpredicates
	 * on the candidate.
	 * 
	 * @param results the results of running each of the subpredicates on the candidate.
	 * @return
	 */
	protected abstract boolean eval(boolean [] results);
	
	public void setSubpredicates(List<FilterPredicate<TINOUT, ?>> subpredicates) throws BadFilterExpressionException {
		this.subpredicates = subpredicates;
	}

	public boolean eval(TINOUT candidate) {
		boolean [] results = new boolean[subpredicates.size()];
		for (int i=0; i<subpredicates.size(); i++) {
			results[i] = subpredicates.get(i).eval(candidate);
		}
		return eval(results);
	}

	public Class<TINOUT> getInputClass() {
		return klass;
	}

	public Class<TINOUT> getOutputClass() {
		return klass;
	}

}
