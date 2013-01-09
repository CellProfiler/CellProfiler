/**
 * 
 */
package org.cellprofiler.imageset;

import static org.junit.Assert.*;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.junit.Test;

/**
 * @author Lee Kamentsky
 *
 */
public class TestRegexpMetadataExtractor {
	private void testSomething(String pattern, String input, String [][] expected) {
		RegexpMetadataExtractor x = new RegexpMetadataExtractor(pattern);
		Map<String, String> map = x.extract(input);
		Set<String> expectedKeys = new HashSet<String>();
		for (String [] kv:expected) {
			assertTrue(map.containsKey(kv[0]));
			assertEquals(kv[1], map.get(kv[0]));
			expectedKeys.add(kv[0]);
		}
		for (String key:map.keySet()) {
			assertTrue(expectedKeys.contains(key));
		}
	}
	
	@Test
	public void testNoKV() {
		testSomething("foo", "foobar", new String [0][]);
	}
	@Test
	public void testNoMatch() {
		testSomething("foo", "bar", new String[0][]);
	}
	@Test
	public void testOneKV() {
		testSomething("_(?P<WellName>[A-Z][0-9]{2})_", "Plate_A01_.png", 
				new String[][] { { "WellName", "A01"}});
		testSomething("_(?P<WellName>[A-Z][0-9]{2})_", "Plate_AXY_.png", new String[0][]);
	}
	@Test
	public void testParseBackslash() {
		testSomething("_\\((?P<WellName>[A-Z][0-9]{2})\\)_", "Plate_(A01)_.png",
				new String[][] { { "WellName", "A01"}});
	}
	@Test
	public void testTwoKV() {
		testSomething("_(?P<WellRow>[A-Z])(?P<WellColumn>[0-9]{2})_", "Plate_A01_.png",
				new String[][] {{"WellRow", "A"}, {"WellColumn", "01"}});
	}
	@Test
	public void testIAmAskingForTrouble() {
		testSomething("_(?P<WellName>[A-Z][0-9]{2})|(?P<FamousCowgirl>Annie +Oakley)_",
				"Plate_A01.jpg", new String [][] { {"WellName", "A01"}});
		testSomething("_(?P<WellName>[A-Z][0-9]{2})|(?P<FamousCowgirl>Annie +Oakley)_",
				"_Annie Oakley_.jpg", new String [][] { {"FamousCowgirl", "Annie Oakley"}});
	}
}
