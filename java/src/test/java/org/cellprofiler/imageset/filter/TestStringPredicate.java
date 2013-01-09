/**
 * 
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
