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

import static org.junit.Assert.*;

import java.io.StringReader;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import javax.json.Json;
import javax.json.JsonObject;
import javax.json.JsonReader;

import org.junit.Test;

/**
 * @author Lee Kamentsky
 *
 * Test the Details class
 */
public class TestDetails {

	/**
	 * Test method for {@link org.cellprofiler.imageset.Details#Details()}.
	 */
	@Test
	public void testDetails() {
		new Details();
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.Details#Details(org.cellprofiler.imageset.Details)}.
	 */
	@Test
	public void testDetailsDetails() {
		new Details(new Details());
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.Details#Details(org.cellprofiler.imageset.Details, org.cellprofiler.imageset.Details)}.
	 */
	@Test
	public void testDetailsDetailsDetails() {
		Details dc = new Details();
		Details dp = new Details();
		dp.put("Foo", "Bar");
		dp.put("Baz", "UnBlech");
		dc.put("Baz", "Blech");
		Details d = new Details(dp, dc);
		assertNotSame(d, dc);
		assertEquals(d.get("Foo"), "Bar");
		assertEquals(d.get("Baz"), "Blech");
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.Details#containsKey(java.lang.String)}.
	 */
	@Test
	public void testContainsKey() {
		Details d = new Details();
		d.put("Foo", "Bar");
		assertTrue(d.containsKey("Foo"));
		assertFalse(d.containsKey("Bar"));
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.Details#get(java.lang.String)}.
	 */
	@Test
	public void testGet() {
		Details d = new Details();
		d.put("Foo", "Bar");
		assertEquals(d.get("Foo"), "Bar");
		assertNull(d.get("Bar"));
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.Details#put(java.lang.String, java.lang.String)}.
	 */
	@Test
	public void testPut() {
		new Details().put("Foo", "Bar");
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.Details#putAll(java.util.Map)}.
	 */
	@Test
	public void testPutAll() {
		Map<String, String> map = new HashMap<String, String>();
		map.put("Foo", "Bar");
		map.put("Baz", "Blech");
		Details d = new Details();
		d.putAll(map);
		assertEquals(d.get("Foo"), "Bar");
		assertEquals(d.get("Baz"), "Blech");
	}

	/**
	 * Test method for {@link org.cellprofiler.imageset.Details#iterator()}.
	 */
	@Test
	public void testIterator() {
		Details d = new Details();
		d.put("Foo", "Bar");
		d.put("Baz", "Blech");
		Iterator<String> id = d.iterator();
		assertTrue(id.hasNext());
		String k1 = id.next();
		assertTrue((k1 == "Foo") || (k1 == "Baz") );
		assertTrue(id.hasNext());
		String k2 = id.next();
		assertTrue((k2 == "Foo") || (k2 == "Baz") );
		assertFalse(k1.equals(k2));
		assertFalse(id.hasNext());
	}		

	/**
	 * Test method for {@link org.cellprofiler.imageset.Details#jsonSerialize()}.
	 */
	@Test
	public void testJsonSerialize() {
		Details dp = new Details();
		dp.put("Foo", "Bar");
		Details d = new Details(dp);
		d.put("Baz", "Blech");
		String s = d.jsonSerialize();
		JsonReader rdr = Json.createReader(new StringReader(s));
		JsonObject o = rdr.readObject();
		assertEquals(o.getString("Foo"), "Bar");
		assertEquals(o.getString("Baz"), "Blech");
	}

}
