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
package org.cellprofiler.headlesspreferences;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.prefs.BackingStoreException;
import java.util.prefs.Preferences;

import org.junit.Test;

/**
 * @author Lee Kamentsky
 * 
 * Tests for the headless preferences factory
 *
 */
public class TestHeadlessPreferences {
	@Test
	public final void testGetSystemRoot() {
		final Preferences root = new HeadlessPreferencesFactory().systemRoot();
		assertNotNull(root);
		assertSame(root, new HeadlessPreferencesFactory().systemRoot());
	}
	
	@Test
	public final void testGetUserRoot() {
		final Preferences root = new HeadlessPreferencesFactory().userRoot();
		assertNotNull(root);
		assertSame(root, new HeadlessPreferencesFactory().systemRoot());
	}

	@Test
	public final void testRemoveNodeSpi() {
		final Preferences root = new HeadlessPreferencesFactory().userRoot();
		final Preferences child = root.node("testRemoveNodeSpi");
		try {
			child.removeNode();
		} catch (BackingStoreException e) {
			fail("Should not throw backing store exception");
		}
		try {
			child.put("Hello", "World");
		} catch (IllegalStateException e) {
			return;
		}
		fail("Should have thrown an IllegalStateException");
	}

	@Test
	public final void testSyncSpi() {
		try {
			new HeadlessPreferencesFactory().userRoot().sync();
		} catch (BackingStoreException e) {
			fail("Should not throw backing store exception");
		}
	}

	@Test
	public final void testFlushSpi() {
		try {
			new HeadlessPreferencesFactory().userRoot().flush();
		} catch (BackingStoreException e) {
			fail("Should not throw backing store exception");
		}
	}

	@Test
	public final void testGetPutSpiStringString() {
		final Preferences root = new HeadlessPreferencesFactory().userRoot();
		final Preferences child = root.node("GetPutSpiStringString");
		child.put("Hello", "World");
		assertEquals("World", child.get("Hello", null));
		assertEquals("Joe", child.get("Goodbye", "Joe"));
	}

	@Test
	public final void testRemoveSpiString() {
		final Preferences root = new HeadlessPreferencesFactory().userRoot();
		final Preferences child = root.node("RemoveSpiString");
		child.put("Hello", "World");
		child.remove("Hello");
		assertEquals("Joe", child.get("Hello", "Joe"));
	}

	@Test
	public final void testKeysSpi() {
		final Preferences root = new HeadlessPreferencesFactory().userRoot();
		final Preferences child = root.node("testKeysSpi");
		child.put("Hello", "World");
		try {
			assertArrayEquals(new String [] {"Hello"}, child.keys());
		} catch (BackingStoreException e) {
			fail("Should not throw backing store exception");
		}
	}

	@Test
	public final void testChildrenNamesSpi() {
		final Preferences root = new HeadlessPreferencesFactory().userRoot();
		final Preferences child = root.node("testChildrenNamesSpi");
		String [] grandchild_names = { "Harry", "May", "Emily" };
		for (String name:grandchild_names) {
			child.node(name);
		}
		try {
			final Set<String> result = new HashSet<String>(Arrays.asList(child.childrenNames()));
			for (String name:grandchild_names) {
				assertTrue(result.contains(name));
			}
		} catch (BackingStoreException e) {
			fail("Should not throw backing store exception");
		}
	}
}
