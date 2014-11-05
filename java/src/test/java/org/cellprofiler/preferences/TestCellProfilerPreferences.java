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
package org.cellprofiler.preferences;

import static org.junit.Assert.*;


import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.WeakHashMap;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.TimeUnit;
import java.util.prefs.BackingStoreException;
import java.util.prefs.NodeChangeEvent;
import java.util.prefs.NodeChangeListener;
import java.util.prefs.PreferenceChangeEvent;
import java.util.prefs.PreferenceChangeListener;
import java.util.prefs.Preferences;

import net.imagej.updater.UpToDate;

import org.cellprofiler.headlesspreferences.HeadlessPreferencesFactory;
import org.junit.Ignore;
import org.junit.Test;

/**
 * @author Lee Kamentsky
 *
 */
public class TestCellProfilerPreferences {
	
	private Preferences getRoot() {
		Preferences root = new HeadlessPreferencesFactory().systemRoot();
		return root;
	}
	
	private Preferences getCPRoot() {
		final WeakHashMap<String, CellProfilerPreferences> map = new WeakHashMap<String, CellProfilerPreferences>();
		final CellProfilerPreferences result = new CellProfilerPreferences(
				null, getRoot(), map);
		map.put("/", result);
		return result;
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#clear()}.
	 */
	@Test
	public void testClear() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		delegate.put("Foo", "Bar");
		try {
			prefs.clear();
			assertEquals("Baz", delegate.get("Foo", "Baz"));
		} catch (BackingStoreException e) {
			e.printStackTrace();
			fail();
		}
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#removeNode()}.
	 */
	@Test
	public void testRemoveNode() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		delegate.node("/testRemoveNode/Foo").put("Bar", "Baz");
		Preferences foo = prefs.node("/testRemoveNode/Foo");
		try {
			assertTrue(delegate.node("/testRemoveNode/Foo").nodeExists(""));
			foo.removeNode();
			assertFalse(delegate.node("/testRemoveNode/Foo").nodeExists(""));
		} catch (BackingStoreException e) {
			e.printStackTrace();
			fail();
		}
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#isUserNode()}.
	 */
	@Test
	public void testIsUserNode() {
		assertFalse(getCPRoot().isUserNode());
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#flush()}.
	 */
	@Test
	public void testFlush() {
		try {
			getCPRoot().flush();
		} catch (BackingStoreException e) {
			e.printStackTrace();
			fail();
		}
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#sync()}.
	 * 
	 * Ignored because there is a bug in Java 1.5 that causes this test to fail
	 * and it's not my fault and I can't be expected to work around it.
	 * 
	 * http://bugs.sun.com/bugdatabase/view_bug.do?bug_id=6178148
	 * JDK-6178148 : (prefs) Preferences.removeNode() bug causes IllegalStateException
	 */
	@Ignore
	@Test
	public void testSync() {
		try {
			getCPRoot().sync();
		} catch (BackingStoreException e) {
			e.printStackTrace();
			fail();
		}
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#getSystemRoot()}.
	 */
	@Test
	public void testGetSystemRoot() {
		Preferences prefs = CellProfilerPreferences.getSystemRoot();
		assertFalse(prefs.isUserNode());
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#getUserRoot()}.
	 */
	@Test
	public void testGetUserRoot() {
		Preferences prefs = CellProfilerPreferences.getUserRoot();
		assertTrue(prefs.isUserNode());
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#put(java.lang.String, java.lang.String)}.
	 */
	@Test
	public void testPutStringString() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		prefs.put("testPutStringString", "Foo");
		assertEquals("Foo", delegate.get("testPutStringString", "Bar"));
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#get(java.lang.String, java.lang.String)}.
	 */
	@Test
	public void testGetStringString() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		delegate.put("testGetStringString", "Foo");
		assertEquals("Foo", prefs.get("testGetStringString", "Bar"));
	}
	
	/**
	 * Make sure that if we get the latest nag time, we'll get
	 * Long.MAX_VALUE
	 */
	@Test
	public void testGetLatestNag() {
		String key = UpToDate.class.getSimpleName() + ".latestNag";
		String path = "/"+UpToDate.class.getPackage().getName().replace(".", "/");
		String result = getCPRoot().node(path).get(key, "0");
		assertEquals(Long.MAX_VALUE, Long.parseLong(result));
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#remove(java.lang.String)}.
	 */
	@Test
	public void testRemoveString() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		delegate.put("testRemoveString", "Foo");
		prefs.remove("testRemoveString");
		assertEquals("Bar", delegate.get("testRemoveString", "Bar"));
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#putInt(java.lang.String, int)}.
	 */
	@Test
	public void testPutIntStringInt() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		prefs.putInt("putIntStringInt", 5);
		assertEquals(5, delegate.getInt("putIntStringInt", 10));
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#getInt(java.lang.String, int)}.
	 */
	@Test
	public void testGetIntStringInt() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		delegate.putInt("getIntStringInt", 7);
		assertEquals(7, prefs.getInt("getIntStringInt", 10));
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#putLong(java.lang.String, long)}.
	 */
	@Test
	public void testPutLongStringLong() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		prefs.putLong("putLongStringLong", 50);
		assertEquals(50L, delegate.getLong("putLongStringLong", 10));
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#getLong(java.lang.String, long)}.
	 */
	@Test
	public void testGetLongStringLong() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		delegate.putLong("getLongStringLong", 70);
		assertEquals(70, prefs.getLong("getLongStringLong", 10));
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#putBoolean(java.lang.String, boolean)}.
	 */
	@Test
	public void testPutBooleanStringBoolean() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		prefs.putBoolean("putBooleanStringBoolean", true);
		assertEquals(true, delegate.getBoolean("putBooleanStringBoolean", false));
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#getBoolean(java.lang.String, boolean)}.
	 */
	@Test
	public void testGetBooleanStringBoolean() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		delegate.putBoolean("getBooleanStringBoolean", false);
		assertEquals(false, prefs.getBoolean("getBooleanStringBoolean", true));
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#putFloat(java.lang.String, float)}.
	 */
	@Test
	public void testPutFloatStringFloat() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		prefs.putFloat("putFloatStringFloat", .7f);
		assertEquals(.7f, delegate.getFloat("putFloatStringFloat", 10), 0.0001);
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#getFloat(java.lang.String, float)}.
	 */
	@Test
	public void testGetFloatStringFloat() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		delegate.putFloat("getFloatStringFloat", .8f);
		assertEquals(.8f, prefs.getFloat("getFloatStringFloat", 10), 0.0001);
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#putDouble(java.lang.String, double)}.
	 */
	@Test
	public void testPutDoubleStringDouble() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		prefs.putDouble("putDoubleStringDouble", .9);
		assertEquals(.9, delegate.getFloat("putDoubleStringDouble", 10), 0.0001);
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#getDouble(java.lang.String, double)}.
	 */
	@Test
	public void testGetDoubleStringDouble() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		delegate.putDouble("getDoubleStringDouble", 1.8);
		assertEquals(1.8, prefs.getDouble("getDoubleStringDouble", 10), 0.0001);
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#putByteArray(java.lang.String, byte[])}.
	 */
	@Test
	public void testPutByteArrayStringByteArray() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		byte [] b = "Foo".getBytes();
		prefs.putByteArray("putByteArray", b);
		assertArrayEquals(b, delegate.getByteArray("putByteArray", "Bar".getBytes()));
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#getByteArray(java.lang.String, byte[])}.
	 */
	@Test
	public void testGetByteArrayStringByteArray() {
		Preferences delegate = getRoot();
		Preferences prefs = getCPRoot();
		byte [] b = "Baz".getBytes();
		delegate.putByteArray("getByteArray", b);
		assertArrayEquals(b, prefs.getByteArray("getByteArray", "Bar".getBytes()));
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#keys()}.
	 */
	@Test
	public void testKeys() {
		Preferences prefs = getCPRoot();
		String path = UpToDate.class.getPackage().getName().replace(".", "/");
		prefs.node(path).put("Foo", "Bar");
		Preferences node = prefs.node(path);
		try {
			List<String> keys = Arrays.asList(node.keys());
			assertTrue(keys.contains("UpToDate.latestNag"));
			assertTrue(keys.contains("Foo"));
		} catch (BackingStoreException e) {
			e.printStackTrace();
			fail();
		}
		
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#childrenNames()}.
	 */
	@Test
	public void testChildrenNames() {
		Preferences prefs = getCPRoot();
		String path = "/" + UpToDate.class.getPackage().getName().replace(".", "/");
		Preferences node = prefs.node(path.substring(0, path.lastIndexOf("/")));
		try {
			List<String> names = Arrays.asList(node.childrenNames());
			assertTrue(names.contains(path.substring(path.lastIndexOf("/")+1)));
			names = Arrays.asList(prefs.childrenNames());
			assertTrue(names.contains(path.substring(1, path.indexOf("/", 1))));
		} catch (BackingStoreException e) {
			e.printStackTrace();
			fail();
		}
		
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#parent()}.
	 */
	@Test
	public void testParent() {
		Preferences child = getCPRoot().node("/Foo/Bar");
		assertEquals("/Foo", child.parent().absolutePath());
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#node(java.lang.String)}.
	 */
	@Test
	public void testNodeString() {
		Preferences node = getCPRoot().node("/Foo/Bar");
		assertEquals("/Foo/Bar", node.absolutePath());
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#nodeExists(java.lang.String)}.
	 */
	@Test
	public void testNodeExistsString() {
		getRoot().node("/testNodeExistsString/Foo/Bar");
		try {
			assertTrue(getCPRoot().nodeExists("/testNodeExistsString/Foo/Bar"));
			assertFalse(getCPRoot().nodeExists("/testNodeExistsString/Bar/Foo"));
		} catch (BackingStoreException e) {
			e.printStackTrace();
			fail();
		}
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#name()}.
	 */
	@Test
	public void testName() {
		assertEquals("Bar", getRoot().node("/testName/Foo/Bar").name());
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#absolutePath()}.
	 */
	@Test
	public void testAbsolutePath() {
		assertEquals("/testAbsolutePath/Foo/Bar", getRoot().node("/testAbsolutePath/Foo/Bar").absolutePath());
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#addPreferenceChangeListener(java.util.prefs.PreferenceChangeListener)}.
	 */
	@Test
	public void testAddPreferenceChangeListenerPreferenceChangeListener() {
		final SynchronousQueue<Boolean> done = new SynchronousQueue<Boolean>();
		final String key = "testAddPreferenceChangeListenerPreferenceChangeListener";
		Preferences prefs = getCPRoot().node("testAddPreferenceChangeListener");
		prefs.addPreferenceChangeListener(new PreferenceChangeListener() {

			public void preferenceChange(PreferenceChangeEvent evt) {
				assertEquals(key, evt.getKey());
				assertEquals("Bar", evt.getNewValue());
				try {
					done.put(true);
				} catch (InterruptedException e) {
					fail();
				}
			}});
		prefs.put(key, "Bar");
		try {
			assertTrue(done.poll(10, TimeUnit.SECONDS));
		} catch (InterruptedException e) {
			e.printStackTrace();
			fail();
		}
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#removePreferenceChangeListener(java.util.prefs.PreferenceChangeListener)}.
	 */
	@Test
	public void testRemovePreferenceChangeListenerPreferenceChangeListener() {
		final SynchronousQueue<Boolean> done = new SynchronousQueue<Boolean>();
		final String key = "testRemovePreferenceChangeListenerPreferenceChangeListener";
		Preferences prefs = getCPRoot().node(key);
		final PreferenceChangeListener pcl = new PreferenceChangeListener() {

			public void preferenceChange(PreferenceChangeEvent evt) {
				assertEquals(key, evt.getKey());
				assertEquals("Bar", evt.getNewValue());
				try {
					done.put(true);
				} catch (InterruptedException e) {
					e.printStackTrace();
					fail();
				}
			}};
		prefs.addPreferenceChangeListener(pcl);
		prefs.put(key, "Bar");
		try {
			assertTrue(done.poll(1, TimeUnit.SECONDS));
		} catch (InterruptedException e) {
			e.printStackTrace();
			fail();
		}
		prefs.removePreferenceChangeListener(pcl);
		prefs.put(key, "Baz");
		try {
			assertNull(done.poll(200, TimeUnit.MILLISECONDS));
		} catch (InterruptedException e) {
			e.printStackTrace();
			fail();
		}
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#addNodeChangeListener(java.util.prefs.NodeChangeListener)}.
	 */
	@Test
	public void testAddNodeChangeListenerNodeChangeListener() {
		final SynchronousQueue<String> done = new SynchronousQueue<String>();
		final String key = "tanclncl";
		Preferences prefs = getCPRoot().node("testAddNodeChangeListenerNodeChangeListener");
		prefs.addNodeChangeListener(new NodeChangeListener() {

			public void childAdded(NodeChangeEvent evt) {
				assertEquals(key, evt.getChild().name());
				try {
					done.put("Added");
				} catch (InterruptedException e) {
					e.printStackTrace();
					fail();
				}
			}

			public void childRemoved(NodeChangeEvent evt) {
				assertEquals(key, evt.getChild().name());
				try {
					done.put("Removed");
				} catch (InterruptedException e) {
					e.printStackTrace();
					fail();
				}
			}});
		Preferences child = prefs.node(key);
		child.put("Foo", "Bar");
		try {
			assertEquals("Added", done.poll(1, TimeUnit.SECONDS));
			child.removeNode();
			assertEquals("Removed", done.poll(1, TimeUnit.SECONDS));
		} catch (BackingStoreException e) {
			e.printStackTrace();
			fail();
		} catch (InterruptedException e1) {
			e1.printStackTrace();
			fail();
		}
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#removeNodeChangeListener(java.util.prefs.NodeChangeListener)}.
	 */
	@Test
	public void testRemoveNodeChangeListenerNodeChangeListener() {
		final SynchronousQueue<String> done = new SynchronousQueue<String>();
		final String key = "testRemoveNodeChangeListenerNodeChangeListener";
		Preferences prefs = getCPRoot().node(key);
		final NodeChangeListener ncl = new NodeChangeListener() {

			public void childAdded(NodeChangeEvent evt) {
				assertEquals(key, evt.getChild().name());
				try {
					done.put("Added");
				} catch (InterruptedException e) {
					e.printStackTrace();
					fail();
				}
			}

			public void childRemoved(NodeChangeEvent evt) {
				fail();
			}};
		prefs.addNodeChangeListener(ncl);
		Preferences child = prefs.node(key);
		try {
			assertEquals("Added", done.poll(10, TimeUnit.SECONDS));
			prefs.removeNodeChangeListener(ncl);
			try {
				child.removeNode();
			} catch (BackingStoreException e) {
				e.printStackTrace();
				fail();
			}
			assertNull(done.poll(100, TimeUnit.MILLISECONDS));
		} catch (InterruptedException e1) {
			e1.printStackTrace();
			fail();
		}
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#exportNode(java.io.OutputStream)}.
	 */
	@Test
	public void testExportNodeOutputStream() {
		ByteArrayOutputStream os1 = new ByteArrayOutputStream();
		ByteArrayOutputStream os2 = new ByteArrayOutputStream();
		getRoot().put("Foo", "Bar");
		try {
			getCPRoot().exportNode(os1);
			getRoot().exportNode(os2);
			assertArrayEquals(os2.toByteArray(), os1.toByteArray());
		} catch (IOException e) {
			e.printStackTrace();
			fail();
		} catch (BackingStoreException e) {
			e.printStackTrace();
			fail();
		}
	}

	/**
	 * Test method for {@link org.cellprofiler.preferences.CellProfilerPreferences#exportSubtree(java.io.OutputStream)}.
	 */
	@Test
	public void testExportSubtreeOutputStream() {
		ByteArrayOutputStream os1 = new ByteArrayOutputStream();
		ByteArrayOutputStream os2 = new ByteArrayOutputStream();
		getRoot().node("testExportSubtreeOutputStream").put("Foo", "Bar");
		try {
			getCPRoot().exportSubtree(os1);
			getRoot().exportSubtree(os2);
			assertArrayEquals(os2.toByteArray(), os1.toByteArray());
		} catch (IOException e) {
			e.printStackTrace();
			fail();
		} catch (BackingStoreException e) {
			e.printStackTrace();
			fail();
		}
	}

}
