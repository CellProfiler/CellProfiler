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
 *
 */
package org.cellprofiler.ijutils;

import org.scijava.Priority;
import org.scijava.log.LogService;
import org.scijava.plugin.Plugin;
import org.scijava.service.Service;

import imagej.platform.DefaultAppEventService;
import imagej.ui.UIService;
import imagej.ui.UserInterface;


/**
 * @author Lee Kamentsky
 *
 * Prevent quit from happening.
 * 
 */
@Plugin(type = Service.class, priority = Priority.HIGH_PRIORITY)
public class CellProfilerAppEventService extends DefaultAppEventService {
	static boolean canQuit = false;
	
	/**
	 * Tell the event service that it's ok to quit on an quit event.
	 */
	public static void allowQuit() {
		canQuit = true;
	}
	
	/**
	 * Tell the event service that it's not OK to quit on a quit event.
	 */
	public static void preventQuit() {
		canQuit = false;
	}
	/* (non-Javadoc)
	 * @see imagej.platform.DefaultAppEventService#quit()
	 */
	@Override
	public void quit() {
		if (canQuit) {
			super.quit();
		} else {
			LogService logService = getContext().getService(LogService.class);
			UIService uiService = getContext().getService(UIService.class);
			if (uiService.isVisible()) {
				UserInterface ui = uiService.getDefaultUI();
				logService.info("Quit action: hide the application frame");
				ui.getApplicationFrame().setVisible(false);
			} else {
				logService.info("Quit action: do nothing");
			}
		}
	}

}
