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

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;

import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;
import org.junit.Test;

/**
 * Test all of the logic predicates
 * 
 * @author Lee Kamentsky
 *
 */
public class TestLogicPredicate {
	static class TrueFalse extends AbstractStringPredicate {
		private final boolean result;
		public TrueFalse(boolean result) {
			this.result = result;
		}
		public String getSymbol() {
			return null;
		}

		public boolean eval(String candidate, String literal) {
			return result;
		}
	}
	/**
	 * Feed results into a logic predicate, testing that it evaluates to the expected result.
	 * @param p the predicate under test
	 * @param subresults the desired evaluation values for the subpredicates
	 * @predicate expected the expected evaluation of the predicate under test
	 *
	 */
	protected void testSomething(LogicPredicate<String> p, boolean [] subresults, boolean expected) {
		List<FilterPredicate<String, ?>> subpredicates = new ArrayList<FilterPredicate<String, ?>>();
		for (boolean subresult:subresults) {
			subpredicates.add(new TrueFalse(subresult));
		}
		try {
			p.setSubpredicates(subpredicates);
		} catch (BadFilterExpressionException e) {
			fail("File predicate takes a subpredicate");
		}
		assertEquals(expected, p.eval(subresults));
	}
	
	@Test
	public void testDoesPredicate() {
		testSomething(new DoesPredicate<String>(String.class), new boolean [] { true }, true);
		testSomething(new DoesPredicate<String>(String.class), new boolean [] { false }, false);
	}

	@Test
	public void testDoesNotPredicate() {
		testSomething(new DoesNotPredicate<String>(String.class), new boolean [] { true }, false);
		testSomething(new DoesNotPredicate<String>(String.class), new boolean [] { false }, true);
	}
	@Test
	public void testAndPredicate() {
		testSomething(new AndPredicate<String>(String.class), new boolean [] { true, true }, true);
		testSomething(new AndPredicate<String>(String.class), new boolean [] { false, true }, false);
		testSomething(new AndPredicate<String>(String.class), new boolean [] { false, false }, false);
	}
	@Test
	public void testOrPredicate() {
		testSomething(new OrPredicate<String>(String.class), new boolean [] { true, true }, true);
		testSomething(new OrPredicate<String>(String.class), new boolean [] { false, true }, true);
		testSomething(new OrPredicate<String>(String.class), new boolean [] { false, false }, false);
	}
	
}
