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

import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;
import org.junit.Test;

/**
 * @author Lee Kamentsky
 * 
 * Test the classes derived from AbstractStringPredicate
 *
 */
public class TestStringPredicate {
	
	private void testSomething(AbstractStringPredicate p, String candidate, String literal, boolean expected) {
		try {
			p.setLiteral(literal);
		} catch (BadFilterExpressionException e) {
			fail("String predicates take literals");
		}
		assertEquals(expected, p.eval(candidate));
	}

	@Test
	public void testContainsPredicate() {
		testSomething(new ContainsPredicate(), "ffooo", "foo", true);
		testSomething(new ContainsPredicate(), "ffooo", "bar", false);
	}
	@Test
	public void testContainsRegexpPredicate() {
		testSomething(new ContainsRegexpPredicate(), "ffooo", "[fgh]o{2}", true);
		testSomething(new ContainsRegexpPredicate(), "ffooo", "[bgh]o{2}", false);
		testSomething(new ContainsRegexpPredicate(), "Channel2-01-A-01.tif", 
				"^Channel2\\-(?P<ImageNumber>.+?)\\-(?P<Row>.+?)\\-(?P<Column>.+?)\\.tif$", true);
		testSomething(new ContainsRegexpPredicate(), "Channel1-01-A-01.tif", 
				"^Channel2\\-(?P<ImageNumber>.+?)\\-(?P<Row>.+?)\\-(?P<Column>.+?)\\.tif$", false);
	}
	@Test
	public void testEndsWithPredicate() {
		testSomething(new EndsWithPredicate(), "ffoo", "foo", true);
		testSomething(new EndsWithPredicate(), "ffoo", "bar", false);	
	}
	@Test
	public void testEqualsPredicate() {
		testSomething(new EqualsPredicate(), "foo", "foo", true);
		testSomething(new EqualsPredicate(), "foo", "bar", false);
	}
	@Test
	public void testStartsWithPredicate() {
		testSomething(new StartsWithPredicate(), "foooo", "foo", true);
		testSomething(new StartsWithPredicate(), "foooo", "bar", false);
	}
}
