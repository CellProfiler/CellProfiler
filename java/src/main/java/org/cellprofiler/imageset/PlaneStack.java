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
package org.cellprofiler.imageset;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import net.imglib2.meta.AbstractTypedSpace;
import net.imglib2.meta.AxisType;
import net.imglib2.meta.TypedAxis;
import net.imglib2.meta.Axes;
import net.imglib2.meta.DefaultTypedAxis;

/**
 * @author Lee Kamentsky
 *
 * A plane stack is a grouping of objects representing 2-D planes
 * such as ImagePlane or ImagePlaneDescriptor. The planes should
 * be of the same dimensions and are tiled over the first two
 * dimensions and stacked over the remaining ones.
 */
public class PlaneStack<T> 
	extends AbstractTypedSpace<TypedAxis>
	implements Iterable<T>
{
	private final int [] dims;
	private final Map<List<Integer>, T> planes;
	static public final TypedAxis[] XYAxes = new TypedAxis [] {
			new DefaultTypedAxis(Axes.X), new DefaultTypedAxis(Axes.Y)
	};
	static public final TypedAxis[] XYCAxes = new TypedAxis [] {
		new DefaultTypedAxis(Axes.X), new DefaultTypedAxis(Axes.Y),
		new DefaultTypedAxis(Axes.CHANNEL)
	};
	/**
	 * Overlapping objects are stored as consecutive planes
	 * in a TIF file and C / Z / T metadata are ignored.
	 */
	static public final String OBJECT_PLANE_AXIS_NAME = "ObjectPlane";
	/**
	 * The axis type for an axis that is a stack of labels matrix
	 * planes for overlapping objects.
	 */
	static public final AxisType OBJECT_PLANE_AXIS_TYPE = Axes.get(OBJECT_PLANE_AXIS_NAME, false);
	/**
	 * The axis type for a stack of overlapping labels planes. 
	 */
	static public final TypedAxis[] XYOAxes = new TypedAxis [] {
		new DefaultTypedAxis(Axes.X), new DefaultTypedAxis(Axes.Y),
		new DefaultTypedAxis(OBJECT_PLANE_AXIS_TYPE)
	};
	/**
	 * Construct a plane stack that can hold just a single
	 * plane.
	 */
	public PlaneStack(T plane) {
		this(XYAxes);
		add(plane, 0, 0);
	}
	/**
	 * Construct an ImageStack with fixed axis definitions. For instance:
	 *     new ImageStack(Axes.X, Axes.Y, Axes.CHANNEL, Axes.Z)
	 *     
	 *  for a 3-d stack of 2D color image planes
	 * 
	 * @param axes the axes that define the stack's dimension identity
	 */
	public PlaneStack(final TypedAxis... axes){
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
	
	/**
	 * @return the total # of planes in the stack
	 */
	public int getPlaneCount() {
		return planes.size();
	}

	public Iterator<T> iterator() {
		return planes.values().iterator();
	}
	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	@Override
	public String toString() {
		StringBuilder result = new StringBuilder(this.getClass().getSimpleName());
		result.append(" Axes: ");
		for (int i=0;i < numDimensions(); i++) {
			if (i > 0) result.append(",");
			result.append(axis(i).type().toString());
		}
		result.append(" Planes: ");
		result.append(planes.toString());
		return result.toString();
	}

}
