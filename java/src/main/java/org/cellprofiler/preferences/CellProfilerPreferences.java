/**
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2015 Broad Institute
 * All rights reserved.
 * 
 * Please see the AUTHORS file for credits.
 * 
 * Website: http://www.cellprofiler.org
 */
package org.cellprofiler.preferences;

import java.io.IOException;
import java.io.OutputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.ServiceLoader;
import java.util.WeakHashMap;
import java.util.prefs.BackingStoreException;
import java.util.prefs.NodeChangeListener;
import java.util.prefs.PreferenceChangeListener;
import java.util.prefs.Preferences;
import java.util.prefs.PreferencesFactory;

import net.imagej.options.OptionsMisc;
import net.imagej.updater.UpToDate;

import org.apache.commons.lang3.SystemUtils;
import org.cellprofiler.headlesspreferences.HeadlessPreferencesFactory;

/**
 * @author Lee Kamentsky
 * 
 * CellProfilerPreferences uses the JRE preferences factory
 * to satisfy almost all preferences requests. It intercepts
 * some user preferences for ImageJ that are inappropriate
 * for its use within CellProfiler.
 *
 */
class CellProfilerPreferences extends Preferences {
	/**
	 * @author Lee Kamentsky
	 *
	 * A preferences override:
	 * Report that the key at a given path has
	 * a particular value.
	 */
	static class PreferenceOverride {
		/**
		 * The path to the preferences node
		 */
		final String path;
		/**
		 * The key at the node to override
		 */
		final String key;
		/**
		 * The value to report as overridden
		 */
		final String value;
		PreferenceOverride(String path, String key, String value) {
			this.path = path;
			this.key = key;
			this.value = value;
		}
	}
	private static final Object lock = new Object();
	private static PreferencesFactory delegatePreferencesFactory = null;
	private static List<PreferenceOverride> overrides = new ArrayList<PreferenceOverride> ();
	private static CellProfilerPreferences systemRoot;
	private static CellProfilerPreferences userRoot;
	private static final WeakHashMap<String, CellProfilerPreferences> systemMap =
		new WeakHashMap<String, CellProfilerPreferences>();
	private static final WeakHashMap<String, CellProfilerPreferences> userMap =
		new WeakHashMap<String, CellProfilerPreferences>();

	private final CellProfilerPreferences parent;
	private final Preferences delegate;
	private final WeakHashMap<String, CellProfilerPreferences> nodeMap;

	{
		addOverride(UpToDate.class, "latestNag", Long.toString(Long.MAX_VALUE));
		addOverride(OptionsMisc.class, "exitWhenQuitting", Boolean.toString(false));
	}
	/**
	 * @return the system root preferences to use in the CellProfilerPreferencesFactory 
	 */
	static Preferences getSystemRoot() {
		synchronized (systemMap) {
			if (! systemMap.containsKey("/")) {
				System.err.println("Accessing system root.");
				Thread.dumpStack();
				systemRoot = new CellProfilerPreferences(
						null, getJREPreferencesFactory().systemRoot(), systemMap);
				systemMap.put("/", systemRoot);
			}
		}
		return systemRoot;
	}
	
	/**
	 * @return the user root preferences to use in the CellProfilerPreferencesFactory
	 */
	static Preferences getUserRoot() {
		synchronized (userMap) {
			if (! userMap.containsKey("/")) {
				System.err.println("Accessing user root.");
				Thread.dumpStack();
				userRoot = new CellProfilerPreferences(
						null, getJREPreferencesFactory().userRoot(), userMap);
				System.err.println("Got user root.");
				assert userRoot.isUserNode();
				userMap.put("/", userRoot);
			}
		}
		return userRoot;
	}
	/**
	 * Lookup or create a CellProfilerPreferences node for a given path.
	 * @param delegate
	 * @param map
	 * @return
	 */
	static private CellProfilerPreferences retrieveNode(
			String path, WeakHashMap<String, CellProfilerPreferences> map) {
		synchronized(map) {
			CellProfilerPreferences result = map.get(path);
			if (result == null) {
				CellProfilerPreferences root = map.get("/");
				final Preferences delegate = root.delegate.node(path);
				final int lastSlash = path.lastIndexOf("/");
				CellProfilerPreferences parent = null;
				if (lastSlash == 0) {
					parent = root;
				} else {
					parent = retrieveNode(path.substring(0, lastSlash), map);
				}
				result = new CellProfilerPreferences(parent, delegate, map);
				map.put(path, result);
			}
			return result;
		}
	}
	/**
	 * Create a CellProfilerPreferences that uses the delegate to handle
	 * most of the work.
	 * 
	 * @param parent the parent node to ours or null if root
	 * @param delegate the delegate node that actually performs the work
	 * @param map the weak hash map holding the cached preferences nodes for this tree
	 */
	CellProfilerPreferences(
			CellProfilerPreferences parent, 
			Preferences delegate,
			WeakHashMap<String, CellProfilerPreferences> map) {
		this.parent = parent;
		this.delegate = delegate;
		this.nodeMap = map;
	}
	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#put(java.lang.String, java.lang.String)
	 */
	@Override
	public void put(String key, String value) {
		delegate.put(key, value);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#get(java.lang.String, java.lang.String)
	 */
	@Override
	public String get(String key, String def) {
		for (PreferenceOverride po:overrides) {
			if (key.equals(po.key) && absolutePath().equals(po.path))
				return po.value;
		}
		return delegate.get(key, def);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#remove(java.lang.String)
	 */
	@Override
	public void remove(String key) {
		delegate.remove(key);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#clear()
	 */
	@Override
	public void clear() throws BackingStoreException {
		delegate.clear();
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#putInt(java.lang.String, int)
	 */
	@Override
	public void putInt(String key, int value) {
		delegate.putInt(key, value);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#getInt(java.lang.String, int)
	 */
	@Override
	public int getInt(String key, int def) {
		return delegate.getInt(key, def);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#putLong(java.lang.String, long)
	 */
	@Override
	public void putLong(String key, long value) {
		delegate.putLong(key, value);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#getLong(java.lang.String, long)
	 */
	@Override
	public long getLong(String key, long def) {
		return delegate.getLong(key, def);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#putBoolean(java.lang.String, boolean)
	 */
	@Override
	public void putBoolean(String key, boolean value) {
		delegate.putBoolean(key, value);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#getBoolean(java.lang.String, boolean)
	 */
	@Override
	public boolean getBoolean(String key, boolean def) {
		return delegate.getBoolean(key, def);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#putFloat(java.lang.String, float)
	 */
	@Override
	public void putFloat(String key, float value) {
		delegate.putFloat(key, value);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#getFloat(java.lang.String, float)
	 */
	@Override
	public float getFloat(String key, float def) {
		return delegate.getFloat(key, def);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#putDouble(java.lang.String, double)
	 */
	@Override
	public void putDouble(String key, double value) {
		delegate.putDouble(key, value);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#getDouble(java.lang.String, double)
	 */
	@Override
	public double getDouble(String key, double def) {
		// TODO Auto-generated method stub
		return delegate.getDouble(key, def);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#putByteArray(java.lang.String, byte[])
	 */
	@Override
	public void putByteArray(String key, byte[] value) {
		delegate.putByteArray(key, value);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#getByteArray(java.lang.String, byte[])
	 */
	@Override
	public byte[] getByteArray(String key, byte[] def) {
		return delegate.getByteArray(key, def);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#keys()
	 */
	@Override
	public String[] keys() throws BackingStoreException {
		final List<String> k = new ArrayList<String>(Arrays.asList(delegate.keys()));
		for (PreferenceOverride po:overrides) {
			if (po.path.equals(absolutePath())) {
				k.add(po.key);
			}
		}
		return k.toArray(new String [0]);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#childrenNames()
	 */
	@Override
	public String[] childrenNames() throws BackingStoreException {
		final List<String> k = new ArrayList<String>(Arrays.asList(delegate.childrenNames()));
		final String absPath = absolutePath();
		for (PreferenceOverride po:overrides) {
			String path = po.path;
			if (path.indexOf(absPath) == 0) {
				while(true) {
					final int slashLoc = path.lastIndexOf('/');
					if (slashLoc <= 0) break;
					String child = path.substring(slashLoc+1);
					path = path.substring(0, slashLoc);
					if (path.equals(absPath)) {
						k.add(child);
						break;
					}
				}
			}
		}
		return k.toArray(new String [0]);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#parent()
	 */
	@Override
	public Preferences parent() {
		return parent;
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#node(java.lang.String)
	 */
	@Override
	public Preferences node(String pathName) {
		Preferences target = delegate.node(pathName); 
		return retrieveNode(target.absolutePath(), this.nodeMap);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#nodeExists(java.lang.String)
	 */
	@Override
	public boolean nodeExists(String pathName) throws BackingStoreException {
		return delegate.nodeExists(pathName);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#removeNode()
	 */
	@Override
	public void removeNode() throws BackingStoreException {
		delegate.removeNode();
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#name()
	 */
	@Override
	public String name() {
		return delegate.name();
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#absolutePath()
	 */
	@Override
	public String absolutePath() {
		return delegate.absolutePath();
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#isUserNode()
	 */
	@Override
	public boolean isUserNode() {
		return nodeMap == userMap;
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#toString()
	 */
	@Override
	public String toString() {
		return delegate.toString();
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#flush()
	 */
	@Override
	public void flush() throws BackingStoreException {
		delegate.flush();
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#sync()
	 */
	@Override
	public void sync() throws BackingStoreException {
		delegate.sync();
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#addPreferenceChangeListener(java.util.prefs.PreferenceChangeListener)
	 */
	@Override
	public void addPreferenceChangeListener(PreferenceChangeListener pcl) {
		delegate.addPreferenceChangeListener(pcl);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#removePreferenceChangeListener(java.util.prefs.PreferenceChangeListener)
	 */
	@Override
	public void removePreferenceChangeListener(PreferenceChangeListener pcl) {
		delegate.removePreferenceChangeListener(pcl);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#addNodeChangeListener(java.util.prefs.NodeChangeListener)
	 */
	@Override
	public void addNodeChangeListener(NodeChangeListener ncl) {
		delegate.addNodeChangeListener(ncl);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#removeNodeChangeListener(java.util.prefs.NodeChangeListener)
	 */
	@Override
	public void removeNodeChangeListener(NodeChangeListener ncl) {
		delegate.removeNodeChangeListener(ncl);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#exportNode(java.io.OutputStream)
	 */
	@Override
	public void exportNode(OutputStream os) throws IOException,
			BackingStoreException {
		delegate.exportNode(os);
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.Preferences#exportSubtree(java.io.OutputStream)
	 */
	@Override
	public void exportSubtree(OutputStream os) throws IOException,
			BackingStoreException {
		delegate.exportSubtree(os);
	}
	
	/**
	 * Get the preferences factory supplied by the JRE or
	 * provided as a service.
	 * 
	 * @return the default preferences factory.
	 */
	static private PreferencesFactory getJREPreferencesFactory() {
		synchronized (lock) {
			if (delegatePreferencesFactory == null) {
				do {
					/*
					 * First, see if there is a PreferencesFactory
					 * provided as a service.
					 */
					final ServiceLoader<PreferencesFactory> pfServiceLoader =
						ServiceLoader.loadInstalled(PreferencesFactory.class);
					final Iterator<PreferencesFactory> pfIter = pfServiceLoader.iterator();
					if (pfIter.hasNext()) {
						delegatePreferencesFactory = pfIter.next();
						break;
					}
					/*
					 * Next, try the WindowsPreferencesFactory if OS is Windows.
					 */
					String pfName = (SystemUtils.IS_OS_WINDOWS)?
							"java.util.prefs.WindowsPreferencesFactory":
							"java.util.prefs.FilePreferencesFactory";
					try {
						Class<?> pfClass = Class.forName("java.util.prefs.WindowsPreferencesFactory", false, null);
						Class<?> pfFuckYou = Class.forName("java.util.prefs.WindowsPreferences", true, null);
						Constructor<?> [] pfConstructors = pfClass.getDeclaredConstructors();
						for (Constructor<?> c:pfConstructors) {
							if (c.getParameterTypes().length == 0) {
								/*
								 * Bad boy - it's package-private AND I CALL IT ANYWAY BAH HA HA HA HA HA HA
								 */
								c.setAccessible(true);
								delegatePreferencesFactory = (PreferencesFactory) c.newInstance(new Object [0]);
								break;
							}
						}
						break;
					} catch (ClassNotFoundException e) {
						e.printStackTrace();
					} catch (SecurityException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					} catch (IllegalArgumentException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					} catch (InstantiationException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					} catch (IllegalAccessException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					} catch (InvocationTargetException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					/*
					 * And as a last resort, there's always our headless
					 * preferences factory.
					 */
					delegatePreferencesFactory = new HeadlessPreferencesFactory();
				} while(false);
			}
		}
		return delegatePreferencesFactory;
		
	}
	/**
	 * Add an override to the override list.
	 * @param c
	 * @param key
	 * @param value
	 */
	private static void addOverride(Class<?> c, String key, String value) {
		String path = "/" + c.getPackage().getName().replace(".", "/");
		String prefKey = c.getSimpleName() + "." + key;
		PreferenceOverride override = new PreferenceOverride(path, prefKey, value);
		overrides.add(override);
	}

}
