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
package org.cellprofiler.ijutils;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Collections;

import imagej.ImageJ;
import imagej.data.Dataset;
import imagej.data.DatasetService;
import imagej.data.autoscale.AutoscaleService;
import imagej.data.display.DataView;
import imagej.data.display.ImageDisplay;
import imagej.data.display.ImageDisplayService;
import imagej.data.display.OverlayService;
import imagej.data.overlay.AbstractROIOverlay;
import imagej.data.overlay.Overlay;
import imagej.data.overlay.RectangleOverlay;
import imagej.display.Display;
import imagej.display.DisplayService;

import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.meta.Axes;
import net.imglib2.meta.AxisType;
import net.imglib2.roi.AbstractRegionOfInterest;
import net.imglib2.roi.RegionOfInterest;
import net.imglib2.type.logic.BitType;

import org.junit.BeforeClass;
import org.junit.Test;

/**
 * @author Lee Kamentsky
 *
 */
public class TestOverlayUtils {
	static ImageJ context;
	@BeforeClass
	public static void setUpClass() {
		context = new ImageJ();
	}
	public static ImageDisplayService getImageDisplayService() {
		return context.imageDisplay();
	}
	public static DisplayService getDisplayService() {
		return context.display();
	}
	public static OverlayService getOverlayService() {
		return context.overlay();
	}
	public static DatasetService getDatasetService() {
		return context.dataset();
	}
	/**
	 * Test method for {@link org.cellprofiler.ijutils.OverlayUtils#extractMask(imagej.data.display.ImageDisplay)}.
	 */
	@Test
	public final void testExtractMask() {
		Dataset dataset = getDatasetService().create(new long [] { 15, 25 }, "Foo", new AxisType [] { Axes.X, Axes.Y }, 8, true, false);
		Display<?> display = getDisplayService().createDisplay(getImageDisplayService().createDataView(dataset));
		assertTrue(display instanceof ImageDisplay);
		ImageDisplay iDisplay = (ImageDisplay)display;
		RectangleOverlay o = new RectangleOverlay(context.getContext());
		o.setOrigin(5, 0);
		o.setOrigin(16, 1);
		o.setExtent(6, 0);
		o.setExtent(7, 1);
		getOverlayService().addOverlays(iDisplay, Collections.singletonList((Overlay)o));
		for (DataView v:iDisplay) {
			v.setSelected(true);
		}
		Img<BitType> mask = OverlayUtils.extractMask(iDisplay);
		assertNotNull(mask);
		assertEquals(mask.dimension(0), 25);
		assertEquals(mask.dimension(1), 15);
		long [] position = new long[2];
		Cursor<BitType> c = mask.cursor();
		/*
		 * Note that axes are reversed.
		 */
		while(c.hasNext()) {
			BitType t = c.next();
			c.localize(position);
			assertEquals(t.get(), position[1] >=5 && position[1] < 11 && position[0] >= 16 && position[0] <23);
		}
	}
	/**
	 * A test of a ROI that has no pixel iterator.
	 */
	@Test
	public final void testNonIterableROI() {
		final RegionOfInterest roi = new AbstractRegionOfInterest(2) {
			private double center_x = 25.0;
			private double center_y = 10.0;
			public boolean contains(double[] position) {
				// Mandelbrot set goes from x = -2.5 to 1, y = -1 to 1
				// so we add 2.5, 1 and multiply each by 10.
				final double x0 = position[0] * 10 + center_x;
				final double y0 = position[1] * 10 + center_y;
				double x = x0;
				double y = y0;
				for (int i=0;i < 100; i++) {
					if (x*x + y*y >= 4) return false;
					final double xnext = x*x - y*y + x0;
					y = 2 * x * y + y0;
					x = xnext;
				}
				return true;
			}

			public void move(double displacement, int d) {
				if (d == 0)
					center_x += displacement;
				else
					center_y += displacement;
			}

			@Override
			protected void getRealExtrema(double[] minima, double[] maxima) {
				minima[0] = center_x - 25.0;
				minima[1] = center_y - 10.0;
				maxima[0] = center_x + 10.0;
				maxima[1] = center_y + 10.0;
			}};
		Overlay o = new AbstractROIOverlay<RegionOfInterest>(context.getContext(), roi) {

			/* (non-Javadoc)
			 * @see imagej.data.overlay.Overlay#move(double[])
			 */
			public void move(double[] deltas) {
				roi.move(deltas);
			}
		};
		Dataset dataset = getDatasetService().create(new long [] { 30, 30 }, "Foo", new AxisType [] { Axes.X, Axes.Y }, 8, true, false);
		Display<?> display = getDisplayService().createDisplay(getImageDisplayService().createDataView(dataset));
		assertTrue(display instanceof ImageDisplay);
		ImageDisplay iDisplay = (ImageDisplay)display;
		getOverlayService().addOverlays(iDisplay, Collections.singletonList((Overlay)o));
		for (DataView v:iDisplay) {
			v.setSelected(true);
		}
		Img<BitType> mask = OverlayUtils.extractMask(iDisplay);
		assertNotNull(mask);
		Cursor<BitType> c = mask.cursor();
		double [] position = new double [2];
		while(c.hasNext()) {
			BitType t = c.next();
			position[1] = c.getDoublePosition(0);
			position[0] = c.getDoublePosition(1);
			assertEquals(t.get(), roi.contains(position));
		}
	}
	@Test
	public void testNoOverlay() {
		// Make sure that extractMask returns null if no overlay.
		Dataset dataset = getDatasetService().create(new long [] { 30, 30 }, "Foo", new AxisType [] { Axes.X, Axes.Y }, 8, true, false);
		Display<?> display = getDisplayService().createDisplay(getImageDisplayService().createDataView(dataset));
		assertTrue(display instanceof ImageDisplay);
		ImageDisplay iDisplay = (ImageDisplay)display;
		for (DataView v:iDisplay) {
			v.setSelected(true);
		}
		assertNull(OverlayUtils.extractMask(iDisplay));
	}
	@Test
	public void testDeselectedOverlay() {
		Dataset dataset = getDatasetService().create(new long [] { 15, 25 }, "Foo", new AxisType [] { Axes.X, Axes.Y }, 8, true, false);
		Display<?> display = getDisplayService().createDisplay(getImageDisplayService().createDataView(dataset));
		assertTrue(display instanceof ImageDisplay);
		ImageDisplay iDisplay = (ImageDisplay)display;
		RectangleOverlay o = new RectangleOverlay(context.getContext());
		o.setOrigin(5, 0);
		o.setOrigin(3, 1);
		o.setExtent(6, 0);
		o.setExtent(7, 1);
		getOverlayService().addOverlays(iDisplay, Collections.singletonList((Overlay)o));
		for (DataView v:iDisplay) {
			v.setSelected(v.getData() != o);
		}
		assertNull(OverlayUtils.extractMask(iDisplay));
	}
	@Test
	public void testTwoOverlays() {
		Dataset dataset = getDatasetService().create(new long [] { 30, 30 }, "Foo", new AxisType [] { Axes.X, Axes.Y }, 8, true, false);
		Display<?> display = getDisplayService().createDisplay(getImageDisplayService().createDataView(dataset));
		assertTrue(display instanceof ImageDisplay);
		ImageDisplay iDisplay = (ImageDisplay)display;
		final ArrayList<Overlay> oo = new ArrayList<Overlay>();
		for (double [][] coords:new double [][][] {
				{ { 5, 6}, { 3, 7 }},
				{ { 1, 4}, { 14, 5}}
		}) {
			final RectangleOverlay o = new RectangleOverlay(context.getContext());
			for (int i=0; i<coords.length; i++) {
				o.setOrigin(coords[i][0], i);
				o.setExtent(coords[i][1], i);
				oo.add(o);
			}
		}
		getOverlayService().addOverlays(iDisplay, oo);
		for (DataView v:iDisplay) {
			v.setSelected(true);
		}
		Img<BitType> mask = OverlayUtils.extractMask(iDisplay);
		assertNotNull(mask);
		assertNotNull(mask);
		Cursor<BitType> c = mask.cursor();
		double [] position = new double [2];
		while(c.hasNext()) {
			BitType t = c.next();
			position[1] = c.getDoublePosition(0);
			position[0] = c.getDoublePosition(1);
			boolean inside = false;
			for (Overlay o:oo) {
				inside |= o.getRegionOfInterest().contains(position);
			}
			assertEquals(t.get(), inside);
		}
	}
}
