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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import net.imglib2.AbstractAnnotatedSpace;
import net.imglib2.Axis;

/**
 * @author Lee Kamentsky
 *
 * A plane stack is a grouping of objects representing 2-D planes
 * such as ImagePlane or ImagePlaneDescriptor. The planes should
 * be of the same dimensions and are tiled over the first two
 * dimensions and stacked over the remaining ones.
 */
public class PlaneStack<T> 
	extends AbstractAnnotatedSpace<Axis>
	implements Iterable<T>
{
	private final int [] dims;
	private final Map<List<Integer>, T> planes;
	/**
	 * Construct an ImageStack with fixed axis definitions. For instance:
	 *     new ImageStack(Axes.X, Axes.Y, Axes.CHANNEL, Axes.Z)
	 *     
	 *  for a 3-d stack of 2D color image planes
	 * 
	 * @param axes the axes that define the stack's dimension identity
	 */
	public PlaneStack(final Axis... axes){
		super(axes);
		dims = new int [axes.length];
		planes = new HashMap<List<Integer>, T>();
	}
	
	/**
	 * Add an image plane to the stack at a given set of coordinates
	 * 
	 * @param plane the plane to be added
	 * @param coords an array of zero-based indexes for each dimension of
	 *        the stack giving the plane's localization within the stack.
	 */
	public void add(T plane, int... coords) {
		final List<Integer> key = makeKey(coords);
		planes.put(key, plane);
		for (int i=0;i < this.numDimensions(); i++) {
			if (coords[i] >= dims[i]) dims[i] = coords[i]+1;
		}
	}

	/**
	 * Make a key into the planes dictionary for the given coordinates
	 * 
	 * @param coords
	 * @return a list suitable as a key
	 */
	static private List<Integer> makeKey(int... coords) {
		final List<Integer> key = new ArrayList<Integer>(coords.length);
		for (int coord:coords) key.add(coord);
		return key;
	}
	
	/**
	 * Get the plane at the given coordinates or null if there
	 * is no plane.
	 * 
	 * @param plane
	 * @param coords
	 * @return
	 */
	public T get(int... coords) {
		final List<Integer> key = makeKey(coords);
		return planes.get(key);
	}
	
	/**
	 * Return the size of the stack in the given dimension.
	 * 
	 * The size is 1+the maximum coordinate in the given dimension
	 * of all of the planes.
	 * 
	 * @param d the dimension along which to retrieve the size.
	 * @return
	 */
	public int size(int d) {
		return dims[d];
	}

	public Iterator<T> iterator() {
		return planes.values().iterator();
	}

}
