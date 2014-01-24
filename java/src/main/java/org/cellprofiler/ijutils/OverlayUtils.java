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

import imagej.data.display.DataView;
import imagej.data.display.ImageDisplay;
import imagej.data.display.OverlayView;
import imagej.data.overlay.Overlay;
import net.imglib2.Cursor;
import net.imglib2.Interval;
import net.imglib2.IterableInterval;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.RealRandomAccess;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.planar.PlanarImgFactory;
import net.imglib2.img.transform.ImgTranslationAdapter;
import net.imglib2.meta.Axes;
import net.imglib2.meta.AxisType;
import net.imglib2.roi.IterableRegionOfInterest;
import net.imglib2.roi.RegionOfInterest;
import net.imglib2.type.logic.BitType;
import net.imglib2.view.Views;

/**
 * @author Lee Kamentsky
 * 
 * This "class" contains Java code to manipulate
 * overlays in ImageJ ImageDisplays
 *
 */
public class OverlayUtils {
	public static Img<BitType> extractMask(ImageDisplay display) {
		Img<BitType> mask = null;
		for (DataView view:display) {
			if (! (view instanceof OverlayView)) continue;
			if (! view.isSelected()) continue;
			OverlayView oView = (OverlayView)view;
			if (mask == null) {
				mask = createBitMask(display);
			}
			Overlay overlay = oView.getData();
			/*
			 * The view can have an offset. Since there are two ways
			 * of handling the ROI, we offset the image instead of the view.
			 * 
			 * We also will attempt to make the axes conform to what's
			 * expected by CP.
			 */
			RandomAccessibleInterval<BitType> adapter = mask;
			long [] offset = new long[display.numDimensions()];
			AxisType [] oAxes = overlay.getAxes();
			AxisType [] dAxes = new AxisType [] { Axes.Y, Axes.X };
				
			for (int i=0; i<oAxes.length; i++) {
				for (int j=0; j<dAxes.length; j++) {
					if (oAxes[i].equals(dAxes[j])) {
						offset[i] = display.getLongPosition(dAxes[j]) - oView.getLongPosition(oAxes[i]);
						if (i != j) {
							/*
							 * Perform an axis permutation.
							 */
							adapter = Views.permute(adapter, j, i);
							final AxisType temp = dAxes[i];
							dAxes[i] = dAxes[j];
							dAxes[j] = temp;
						}
						break;
					}
					
				}
			}
			adapter = Views.translate(adapter, offset); 
			RegionOfInterest roi = overlay.getRegionOfInterest();
			if (roi instanceof IterableRegionOfInterest) {
				/*
				 * Yay! We can iterate over the pixels to turn each of them on.
				 */
				IterableInterval<BitType> ii = 
					((IterableRegionOfInterest) roi).getIterableIntervalOverROI(adapter);
				Cursor<BitType> c = ii.cursor();
				while(c.hasNext()){
					c.next().set(true);
				}
			} else {
				/*
				 * Boo! We have to sample from the ROI.
				 */
				RealRandomAccess<BitType> roiAccess = roi.realRandomAccess();
				Cursor<BitType> c = Views.iterable(adapter).cursor();
				while(c.hasNext()) {
					BitType t = c.next();
					roiAccess.setPosition(c);
					t.set(roiAccess.get().get());
				}
			}
		}
		return mask;
	}
	
	/**
	 * Create a bit type image that covers the given interval
	 * 
	 * @param interval - an interval, typically that of a display
	 * 
	 * @return a bit image of the interval, initialized to zero.
	 * 
	 */
	public static Img<BitType> createBitMask(ImageDisplay interval) {
		long [] dimensions = new long [] { 
				interval.max(interval.getAxisIndex(Axes.Y))+1,
				interval.max(interval.getAxisIndex(Axes.X))+1
		};
		return ArrayImgs.bits(dimensions);
	}

}
