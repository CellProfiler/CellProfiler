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
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import net.imglib2.meta.Axes;
import net.imglib2.meta.TypedAxis;
import ome.xml.model.Image;

import org.cellprofiler.imageset.filter.Filter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Lee Kamentsky
 * 
 * A channel filter defines how images are
 * assembled into a stack and which ones get to
 * participate
 *
 */
public class ChannelFilter {
	/**
	 * @author Lee Kamentsky
	 *
	 * An image plane details comparator that orders by
	 * URI and series. Override to order planes within the same series.  
	 */
	static abstract public class OrderingBase 
		implements Comparator<ImagePlaneDetails>{

		/**
		 * Compare two planes on the basis of their URI and series
		 * @param o1
		 * @param o2
		 * @return negative or positive if the planes are in different series
		 *         or zero if they are within the same series and need further
		 *         comparison. 
		 */
		public int compare(ImagePlaneDetails a, ImagePlaneDetails b) {
			final ImagePlane planeA = a.getImagePlane();
			final ImagePlane planeB = b.getImagePlane();
			int result = planeA.getImageFile().getURI().compareTo(
					planeB.getImageFile().getURI());
			if (result != 0) return result;
			final ImageSeries seriesA = planeA.getSeries();
			final ImageSeries seriesB = planeB.getSeries();
			result = seriesA.getSeries() - seriesB.getSeries();
			if (result != 0) return result;
			final Image omeImageA = seriesA.getOMEImage();
			final Image omeImageB = seriesB.getOMEImage();
			if (omeImageA == null) return (omeImageA==null)?0:-1;
			if (omeImageB == null) return 1;
			return compare(planeA, planeB);
		}
		/**
		 * The implementation-dependent ordering of image planes
		 * based on their OME XML metadata or additional image plane info
		 * 
		 * @param aImagePlane
		 * @param aOMEPlane
		 * @param bImagePlane
		 * @param bOMEPlane
		 * @return
		 */
		abstract protected int compare(ImagePlane aImagePlane, ImagePlane bImagePlane);
		
	}
	/**
	 * @author Lee Kamentsky
	 * An ordering of IPDs by URL / series # / T / Z and C
	 * so that planes of consecutive color indices are
	 * adjacent after ordering.
	 */
	static public class ColorOrdering extends OrderingBase {
		public int compare(ImagePlane aImagePlane, ImagePlane bImagePlane) {
			int result = aImagePlane.theT() - bImagePlane.theT();
			if (result != 0) return result;
			result = aImagePlane.theZ() - bImagePlane.theZ();
			if (result != 0) return result;
			return aImagePlane.theC() - bImagePlane.theC();
		}
	}
	/**
	 * @author Lee Kamentsky
	 *
	 * An ordering of IPDs by URL / series # and index
	 * so that planes of the same URL and series are
	 * ordered by index.
	 */
	static public class ObjectsOrdering extends OrderingBase {

		@Override
		protected int compare(ImagePlane aImagePlane, ImagePlane bImagePlane) {
			return aImagePlane.getIndex() - bImagePlane.getIndex();
		}
		
	}
	/**
	 * @author Lee Kamentsky
	 * 
	 * The default ordering of stacks - 
	 */
	static public class DefaultOrdering implements Comparator<ImagePlaneDetailsStack> {
		private Comparator<ImagePlaneDetails> ipdComparator = new ColorOrdering();

		public int compare(ImagePlaneDetailsStack o1, ImagePlaneDetailsStack o2) {
			return ipdComparator.compare(o1.iterator().next(), o2.iterator().next());
		}
		
	}
	/**
	 * The filter used for filtering ipds
	 */
	final private Filter<ImagePlaneDetailsStack> filter;
	
	final private String name;
	
	final private TypedAxis [] axes;

	private static final String SHORT_IMAGESET_MISSING_MSG = "Image missing from channel";
	private static final Logger LOGGER = LoggerFactory.getLogger(ChannelFilter.class);
	/**
	 * Initialize the channel filter with a filter and metadata keys.
	 * 
	 * @param channelName the name of the channel
	 * @param filter filter for choosing which stacks belong to the channel
	 * @param axes the axes for the filter. Typical choices:
	 *        Monochrome: PlaneStack.XYAxes
	 *        Color: PlaneStack.XYCAxes
	 */
	public ChannelFilter(
			String channelName, 
			Filter<ImagePlaneDetailsStack> filter,
			TypedAxis [] axes) {
		this.name = channelName;
		this.filter = filter;
		this.axes = axes;
	}
	
	/**
	 * Make a channel filter that... doesn't filter.
	 * The purpose of this channel filter is to group whatever
	 * image planes it's given into stacks.
	 * 
	 * @param channelName
	 * @param axes
	 */
	public ChannelFilter(String channelName, TypedAxis []axes) {
		this(channelName, null, axes);
	}
	
	/**
	 * @return this channel filter's filter
	 */
	public Filter<ImagePlaneDetailsStack> getFilter() { return filter; }
	
	/**
	 * @return the channel's name
	 */
	public String getName() { return name; }
	
	/**
	 * @return the axes for the filter
	 */
	public TypedAxis [] getAxes() { return axes; }
	
	/**
	 * Make a list of stacks of image planes from
	 * a list of image plane descriptors
	 * 
	 * @param ipds - the available image plane descriptors
	 * 
	 * @return a list of stacks that pass the filter and 
	 *         are assembled according to the axis shape.
	 */
	public List<ImagePlaneDetailsStack> makeStacks(List<ImagePlaneDetails> ipds) {
		List<ImagePlaneDetailsStack> stacks = getStacks(ipds);
		if (filter == null) return stacks;
		List<ImagePlaneDetailsStack> filteredStacks = new ArrayList<ImagePlaneDetailsStack>();
		for (ImagePlaneDetailsStack stack:stacks) {
			if (filter.eval(stack)) {
				filteredStacks.add(stack);
			}
		}
		return filteredStacks;
	}
	
	/**
	 * Make image sets by order
	 * 
	 * @param channels use these channel filters to make stacks out of ipds
	 * @param ipds the image planes to be dealt to the different channels
	 * @param errors report any per-imageset errors back into this array
	 * @param ordering use this ordering criterion to order the stacks
	 *        (see DefaultOrdering).
	 * @return stacks organized into image sets
	 */
	public static List<ImageSet> makeImageSets(
			List<ChannelFilter> channels, 
			List<ImagePlaneDetails> ipds, 
			List<ImageSetError> errors,
			Comparator<ImagePlaneDetailsStack> ordering) {
		
		final List<List<ImagePlaneDetailsStack>> channelStacks = 
			new ArrayList<List<ImagePlaneDetailsStack>>(channels.size());
		int nImageSets = Integer.MAX_VALUE;
		int nMaxImageSets = 0;
		
		for (ChannelFilter channel:channels) {
			final List<ImagePlaneDetailsStack> stacks = channel.makeStacks(ipds);
			Collections.sort(stacks, ordering);
			channelStacks.add(stacks);
			nImageSets = Math.min(nImageSets, stacks.size());
			nMaxImageSets = Math.max(nImageSets, stacks.size());
		}

		final List<ImageSet> result = new ArrayList<ImageSet>(nImageSets);
		if (nMaxImageSets == 0) {
			LOGGER.warn("Empty image set list: no images passed the filtering criteria.");
			return result;
		}
		final int nDigits = (int) Math.log10(nMaxImageSets) + 1;
		final String keyFormat = String.format("%%0%dd", nDigits);
		//
		// Compile the "missing" errors.
		//
		if (nImageSets < nMaxImageSets) {
			LOGGER.warn("Channels have different numbers of images:");
			ChannelFilter badChannel = null;
			for (int i=0; i<channels.size(); i++) {
				final int nImages = channelStacks.get(i).size();
				final ChannelFilter channelFilter = channels.get(i);
				if (nImages == nImageSets) badChannel = channelFilter;
				final String msgFormat = (nImages == 1)?
						"    Channel %s: %d image":"    Channel %s: %d images";
				LOGGER.warn(String.format(msgFormat, 
						channelFilter.getName(), 
						nImages));
			}
			for (int i=nImageSets; i<nMaxImageSets; i++) {
				// TODO: If someone wants to invest time to make
				//       these error messages more informative,
				//       while considering that a user might
				//       misconfigure and get 1,000,000 of them...
				errors.add(new ImageSetMissingError(
						badChannel.getName(),
						SHORT_IMAGESET_MISSING_MSG,
						Collections.singletonList(String.format(keyFormat, i+1))));
			}
		}
		for (int i=0; i<nImageSets; i++) {
			final ArrayList<ImagePlaneDetailsStack> stacks =
				new ArrayList<ImagePlaneDetailsStack>(channels.size());
			for (int j=0; j<channels.size(); j++) {
				stacks.add(channelStacks.get(j).get(i));
			}
			result.add(new ImageSet(
					stacks, Collections.singletonList(String.format(keyFormat, i+1))));
		}
		return result;
	}
	
	/**
	 * Make image sets using the default stack ordering
	 * 
	 * @param channels
	 * @param ipds
	 * @param errors
	 * @return
	 */
	public static List<ImageSet> makeImageSets(
			List<ChannelFilter> channels, 
			List<ImagePlaneDetails> ipds, 
			List<ImageSetError> errors) {
		return makeImageSets(channels, ipds, errors, new DefaultOrdering());
	}
	/**
	 * Assemble image plane details into plane stacks
	 * @param ipds
	 * @return
	 */
	private List<ImagePlaneDetailsStack> getStacks(List<ImagePlaneDetails> ipds) {
		for (TypedAxis axis: axes) {
			if (axis.type().equals(Axes.CHANNEL)) {
				return getColorStacks(ipds);
			} else if (axis.type().equals(PlaneStack.OBJECT_PLANE_AXIS_TYPE)) {
				return getObjectStacks(ipds);
			}
		}
		return getMonochromeStacks(ipds);
	}

	/**
	 * Deal IPDs into single-plane stacks
	 * @param ipds
	 * @return
	 */
	private List<ImagePlaneDetailsStack> getMonochromeStacks(List<ImagePlaneDetails> ipds) {
		List<ImagePlaneDetailsStack> result = new ArrayList<ImagePlaneDetailsStack>();
		for (ImagePlaneDetails ipd:ipds) {
			result.add(ImagePlaneDetailsStack.makeMonochromeStack(
					ipd.coerceToMonochrome()));
		}
		return result;
	}

	private List<ImagePlaneDetailsStack> getColorStacks(List<ImagePlaneDetails> ipds) {
		ipds = new ArrayList<ImagePlaneDetails>(ipds);
		Collections.sort(ipds, new ColorOrdering());
		List<ImagePlaneDetailsStack> result = new ArrayList<ImagePlaneDetailsStack>();
		//
		// Separate IPDs with decent OME metadata from ones without.
		//
		List<ImagePlaneDetails> groupingPlanes = new ArrayList<ImagePlaneDetails>();
		for (ImagePlaneDetails ipd:ipds) {
			final ImagePlane imagePlane = ipd.getImagePlane();
			if (imagePlane.hasCZT()) {
				groupingPlanes.add(ipd);
			} else {
				result.add(ImagePlaneDetailsStack.makeColorStack(ipd.coerceToColor()));
			}
		}
		if (groupingPlanes.size() > 0) {
			// Order the planes so that consecutive channels of the
			// same channel-stack appear consecutively.
			//
			// We keep adding channels as long as the channel #
			// is larger than the last one we saw. If the new
			// channel # is smaller, that's a signal that we've
			// moved onto the next color stack.
			//
			ImagePlaneDetailsStack stack = new ImagePlaneDetailsStack(PlaneStack.XYCAxes);
			int lastChannel = -1;
			for (ImagePlaneDetails ipd:groupingPlanes) {
				final Integer c = ipd.getImagePlane().theC();
				if (c > lastChannel) {
					stack.add(ipd, 0, 0, c);
				} else {
					result.add(stack);
					stack = ImagePlaneDetailsStack.makeColorStack(ipd);
				}
				lastChannel = c;
			}
			result.add(stack);
		}
		return result;
	}
	private List<ImagePlaneDetailsStack> getObjectStacks(List<ImagePlaneDetails> ipds) {
		ipds = new ArrayList<ImagePlaneDetails>(ipds);
		Collections.sort(ipds, new ObjectsOrdering());
		List<ImagePlaneDetailsStack> result = new ArrayList<ImagePlaneDetailsStack>();
		//
		// Make ChannelGrouping keys for planes that have decent OME metadata
		//
		List<ImagePlaneDetails> groupingPlanes = new ArrayList<ImagePlaneDetails>();
		for (ImagePlaneDetails ipd:ipds) {
			final ImagePlane imagePlane = ipd.getImagePlane();
			if (imagePlane.getOMEPlane() != null) {
				groupingPlanes.add(ipd);
			} else {
				result.add(ImagePlaneDetailsStack.makeObjectsStack(ipd));
			}
		}
		if (groupingPlanes.size() > 0) {
			// Order the planes so that consecutive channels of the
			// same channel-stack appear consecutively.
			//
			// We keep adding channels as long as the channel #
			// is larger than the last one we saw. If the new
			// channel # is smaller, that's a signal that we've
			// moved onto the next color stack.
			//
			ImagePlaneDetailsStack stack = new ImagePlaneDetailsStack(PlaneStack.XYOAxes);
			int lastIndex = -1;
			for (ImagePlaneDetails ipd:groupingPlanes) {
				final int index = ipd.getImagePlane().getIndex();
				if (index > lastIndex) {
					stack.add(ipd, 0, 0, index);
				} else {
					result.add(stack);
					stack = ImagePlaneDetailsStack.makeObjectsStack(ipd);
				}
				lastIndex = index;
			}
			result.add(stack);
		}
		return result;
	}
}