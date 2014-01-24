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
package org.cellprofiler.headlesspreferences;

import java.util.HashMap;
import java.util.Map;
import java.util.prefs.AbstractPreferences;
import java.util.prefs.BackingStoreException;
import java.util.prefs.Preferences;
import java.util.prefs.PreferencesFactory;

/**
 * @author Lee Kamentsky
 *
 * CellProfiler in headless mode isolates itself from user preferences
 * by limiting the scope of changes made to the preferences to the
 * running process. The HeadlessPreferencesFactory maintains a single
 * set of preferences for both the system and user and these preferences
 * start fresh with each invocation of the JVM.
 * 
 * Preferences are used extensively by ImageJ - so this is for ImageJ's
 * benefit.
 * 
 * The idea for this class is largely taken from a post by Rob Slifka
 * in his blog, "All about balance":
 * 
 * http://www.allaboutbalance.com/articles/disableprefs/
 */
public class HeadlessPreferencesFactory implements PreferencesFactory {
	/**
	 * @author Lee Kamentsky
	 * 
	 * An implementation of Preferences that is local to the JVM instance.
	 *
	 * Note that all methods ending in Spi are called with AbstractPreferences
	 *  synchronizing lock.
	 */
	static class HeadlessPreferences extends AbstractPreferences {
		private final Map<String, String> d = new HashMap<String, String>();
		private final Map<String, HeadlessPreferences> children = new HashMap<String, HeadlessPreferences>();
		private boolean alive = true;
		/**
		 * Constructor - pass arguments on to base class
		 * 
		 * @param parent
		 * @param name
		 */
		protected HeadlessPreferences(AbstractPreferences parent, String name) {
			super(parent, name);
		}
		
		private void checkAlive() {
			if (alive == false) {
				throw new IllegalStateException("This preferences node has been removed");
			}
		}

		@Override
		protected void putSpi(String key, String value) {
			checkAlive();
			d.put(key, value);
		}

		@Override
		protected String getSpi(String key) {
			checkAlive();
			return d.get(key);
		}

		@Override
		protected void removeSpi(String key) {
			checkAlive();
			d.remove(key);
		}

		@Override
		protected void removeNodeSpi() throws BackingStoreException {
			alive = false;
			d.clear();
		}

		@Override
		protected String[] keysSpi() throws BackingStoreException {
			checkAlive();
			return d.keySet().toArray(new String [] {});
		}

		@Override
		protected String[] childrenNamesSpi() throws BackingStoreException {
			checkAlive();
			return children.keySet().toArray(new String [] {});
		}

		@Override
		protected AbstractPreferences childSpi(String name) {
			checkAlive();
			if (! children.containsKey(name)) {
				final HeadlessPreferences result = new HeadlessPreferences(this, name); 
				children.put(name, result);
				result.newNode = true;
				return result;
			} else {
				return children.get(name);
			}
		}

		@Override
		protected void syncSpi() throws BackingStoreException {
			// No outside access, nothing to do
		}

		@Override
		protected void flushSpi() throws BackingStoreException {
			// No outside access, nothing to do
		}
		
	}
	static final HeadlessPreferences root = new HeadlessPreferences(null, "");
	/* (non-Javadoc)
	 * @see java.util.prefs.PreferencesFactory#systemRoot()
	 */
	public Preferences systemRoot() {
		return root;
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.PreferencesFactory#userRoot()
	 */
	public Preferences userRoot() {
		return root;
	}

}
