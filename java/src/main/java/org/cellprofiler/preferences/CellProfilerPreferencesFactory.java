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

import java.util.prefs.Preferences;
import java.util.prefs.PreferencesFactory;

/**
 * @author Lee Kamentsky
 *
 */
public class CellProfilerPreferencesFactory implements PreferencesFactory {

	/* (non-Javadoc)
	 * @see java.util.prefs.PreferencesFactory#systemRoot()
	 */
	public Preferences systemRoot() {
		return CellProfilerPreferences.getSystemRoot();
	}

	/* (non-Javadoc)
	 * @see java.util.prefs.PreferencesFactory#userRoot()
	 */
	public Preferences userRoot() {
		return CellProfilerPreferences.getUserRoot();
	}

}
