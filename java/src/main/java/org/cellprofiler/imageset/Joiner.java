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
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

import net.imglib2.meta.Axes;
import net.imglib2.meta.TypedAxis;
import ome.xml.model.Plane;

import org.cellprofiler.imageset.filter.Filter;

/**
 * @author Lee Kamentsky
 * 
 * The joiner is the machine that creates image sets
 * from image plane descriptors.
 *
 */
public class Joiner {
	/**
	 * @author Lee Kamentsky
	 * 
	 * A JoinerException indicates that there is some logical problem
	 * in the way the Joiner was put together, for instance that
	 * it specifies a Cartesian product.
	 *
	 */
	public static class JoinerException extends Exception {
		/**
		 * 
		 */
		private static final long serialVersionUID = 3466971897445983373L;

		public JoinerException(String message) {
			super(message);
		}
		
	}
	/**
	 * @author Lee Kamentsky
	 * 
	 * A channel filter defines a channel's join criteria
	 * and IPD filter.
	 *
	 */
	public static class ChannelFilter {
		/**
		 * The metadata keys used for matching
		 */
		final private String [] keys;
		/**
		 * The filter used for filtering ipds
		 */
		final private Filter<ImagePlaneDetailsStack> filter;
		
		final private String name;
		
		final private TypedAxis [] axes;
		/**
		 * Initialize the channel filter with a filter and metadata keys.
		 * 
		 * @param channelName
		 * @param keys
		 * @param isNumeric - true if the key's value is treated as numeric during matching and sorting
		 * @param filter
		 * @param axes the axes for the filter. Typical choices:
		 *        Monochrome: PlaneStack.XYAxes
		 *        Color: PlaneStack.XYCAxes
		 */
		public ChannelFilter(
				String channelName, 
				String [] keys,
				Filter<ImagePlaneDetailsStack> filter,
				TypedAxis ... axes) {
			this.name = channelName;
			this.keys = keys;
			this.filter = filter;
			this.axes = axes;
		}
		
		/**
		 * Initialize the channel filter to match by order.
		 * 
		 * @param channelName
		 * @param filter
		 */
		public ChannelFilter(String channelName, Filter<ImagePlaneDetailsStack> filter, TypedAxis... axes) {
			this(channelName, null, filter, axes);
		}
		
		public Filter<ImagePlaneDetailsStack> getFilter() { return filter; }
		
		public String getName() { return name; }
		
		public String [] getKeys() { return keys; }
		
		public TypedAxis [] getAxes() { return axes; }
		
		/**
		 * @return the indices of the keys that have non-null metadata tags.
		 */
		public int [] getJoiningKeyIndices() {
			int [] nonNullTags = new int[keys.length];
			int idx = 0;
			for (int i=0; i<keys.length; i++) {
				if (keys[i] != null) {
					nonNullTags[idx++] = i;
				}
			}
			return Arrays.copyOf(nonNullTags, idx);
		}
	}
	
	final private List<ChannelFilter> channels;
	final private int anchorChannel;
	final private int nKeys;
	final private boolean [] isNumeric;
	/**
	 * Create a joiner from a list of channel filters
	 * 
	 * @param channels channel filters that pick and assemble the image stacks in a channel
	 * @param isNumeric - for each matching key, true if metadata values should be matched and sorted
	 *                    numerically, false to match and sort as a string.
	 * @throws JoinerException
	 */
	public Joiner(List<ChannelFilter> channels, boolean [] isNumeric) throws JoinerException {
		if (channels.size() == 0) {
			throw new JoinerException("The image set must have at least one channel");
		}
		this.channels = new ArrayList<ChannelFilter>(channels);
		this.isNumeric = isNumeric;
		int anchorChannel = -1;
		nKeys = channels.get(0).getKeys().length;
		for (int i=0; i<channels.size(); i++) {
			ChannelFilter channel = channels.get(i);
			String [] keys = channel.getKeys();
			if (keys.length != nKeys) {
				throw new JoinerException("All channels must have the same number of keys");
			}
			if (channel.getJoiningKeyIndices().length == nKeys) {
				anchorChannel = i;
			}
		}
		if (anchorChannel == -1) {
			throw new JoinerException("At least one of the channels must have all matching tags specified");
		}
		this.anchorChannel = anchorChannel;
	}
	
	static private <T> List<T> getListOfNulls(int nElements) {
		List<T> result = new ArrayList<T>(nElements);
		for (int i=0; i<nElements; i++) result.add(null);
		return result;
	}
	/**
	 * @author Lee Kamentsky
	 *
	 * A class to compare channel key sets against each other
	 * 
	 * A null key matches anything.
	 * Other keys are matched on either a numeric or string basis
	 * depending on whether the key is interpreted numerically or not.
	 */
	protected class MetadataComparator implements Comparator<List<String>> {

		public int compare(List<String> o1, List<String> o2) {
			for (int i=0; i<isNumeric.length; i++) {
				final String s1 = o1.get(i);
				if (s1 == null) continue;
				final String s2 = o2.get(i);
				if (s2 == null) continue;
				final int cmp = (isNumeric[i])?
						Double.valueOf(s1).compareTo(Double.valueOf(s2)):
						s1.compareTo(s2);
				if (cmp != 0) return cmp;
			}
			return 0;
		}
		
	}
	/**
	 * Join the image plane descriptors to make an image set list
	 * @param ipds image plane descriptors (= image planes + metadata)
	 * @param errors on input, this should be an empty collection. 
	 *        On output it will hold an ImageSetError for every image set key
	 *        which had missing or duplicate images.
	 * @return a list of image sets.
	 */
	public List<ImageSet> join(List<ImagePlaneDetails> ipds, Collection<ImageSetError> errors) {
		SortedMap<List<String>, List<ImagePlaneDetailsStack>> result = 
			new TreeMap<List<String>, List<ImagePlaneDetailsStack>>(new MetadataComparator());
		/*
		 * Do the anchor channel first.
		 */
		Map<List<String>, ImagePlaneDetailsStack> channelResult = filterChannel(
				channels.get(anchorChannel), ipds, errors);
		for (Map.Entry<List<String>, ImagePlaneDetailsStack> entry:channelResult.entrySet()) {
			List<ImagePlaneDetailsStack> imagesetIPDs = getListOfNulls(channels.size());
			imagesetIPDs.set(anchorChannel, entry.getValue());
			result.put(entry.getKey(), imagesetIPDs);
		}
		/*
		 * Loop through the other channels.
		 */
		for (int i=0; i<channels.size(); i++) {
			if (i == anchorChannel) continue;
			final ChannelFilter channelFilter = channels.get(i);
			channelResult = filterChannel(channelFilter, ipds, errors);
			int [] idxs = channelFilter.getJoiningKeyIndices(); 
			List<String> key = getListOfNulls(idxs.length);
			Set<List<String>> channelResultKeys = new HashSet<List<String>>(channelResult.keySet());
			for (Map.Entry<List<String>, List<ImagePlaneDetailsStack>> entry:result.entrySet()) {
				/*
				 * Extract only the metadata values in the result that are specified
				 * by this channel.
				 */
				for (int j=0; j<idxs.length; j++) {
					key.set(j, entry.getKey().get(idxs[j]));
				}
				if (! channelResult.containsKey(key)) {
					ImageSetMissingError error = new ImageSetMissingError(
							channelFilter.getName(), "No matching metadata", entry.getKey()); 
					errors.add(error);
				} else {
					entry.getValue().set(i, channelResult.get(key));
					channelResultKeys.remove(key);
				}
			}
			for (List<String> missingKey:channelResultKeys) {
				ImageSetMissingError error = new ImageSetMissingError(
						channels.get(anchorChannel).getName(),
						"No matching metadata", missingKey);
				final List<ImagePlaneDetailsStack> errantImageSet = new ArrayList<ImagePlaneDetailsStack>(channels.size());
				for (int ii=0; ii<channels.size(); ii++) {
					errantImageSet.add((ii==i)?channelResult.get(missingKey):null);
				}
				error.setImageSet(errantImageSet);
				errors.add(error);
			}
		}
		/*
		 * Remove any image sets with errors
		 */
		for (ImageSetError error: errors) {
			final List<String> key = error.getKey();
			if (result.containsKey(key)) {
				List<ImagePlaneDetailsStack> imageSet = result.get(key);
				error.setImageSet(imageSet);
				result.remove(key);
			}
		}
		/*
		 * Construct the final list
		 */
		List<ImageSet> imageSet = new ArrayList<ImageSet>(result.size());
		for (Map.Entry<List<String>, List<ImagePlaneDetailsStack>> entry:result.entrySet()) {
			imageSet.add(new ImageSet(entry.getValue(), entry.getKey()));
		}
		return imageSet;
	}
	
	private Map<List<String>, ImagePlaneDetailsStack> filterChannel(
			ChannelFilter channelFilter,
			List<ImagePlaneDetails> ipds,
			Collection<ImageSetError> errors) {
		Map<List<String>, ImageSetDuplicateError> localErrors = 
			new HashMap<List<String>, ImageSetDuplicateError>();
		Map<List<String>, ImagePlaneDetailsStack> result = 
			new HashMap<List<String>, ImagePlaneDetailsStack>();

		List<ImagePlaneDetailsStack> ipdStacks = getStacks(ipds, channelFilter.getAxes());
		String [] keys = channelFilter.getKeys();
		int [] idxs = channelFilter.getJoiningKeyIndices();
		next_ipd:
		for (ImagePlaneDetailsStack ipdStack:ipdStacks) {
			if (channelFilter.getFilter().eval(ipdStack)) {
				List<String> keyValues = new ArrayList<String>(idxs.length);
				for (int i:idxs) {
					if (! ipdStack.containsKey(keys[i])) continue next_ipd;
					keyValues.add(ipdStack.get(keys[i]));
				}
				if (result.containsKey(keyValues)) {
					if (localErrors.containsKey(keyValues)) {
						localErrors.get(keyValues).getImagePlaneDetailsStacks().add(ipdStack);
					} else {
						List<ImagePlaneDetailsStack> errorIPDs = new ArrayList<ImagePlaneDetailsStack>(2);
						errorIPDs.add(result.get(keyValues));
						errorIPDs.add(ipdStack);
						localErrors.put(keyValues, 
								new ImageSetDuplicateError(
										channelFilter.getName(), 
										"Duplicate entries for key", 
										keyValues, errorIPDs));
					}
					continue;
				}
				result.put(keyValues, ipdStack);
			}
		}
		errors.addAll(localErrors.values());
		return result;
	}

	private List<ImagePlaneDetailsStack> getStacks(List<ImagePlaneDetails> ipds, TypedAxis [] axes) {
		for (TypedAxis axis: axes) {
			if (axis.type().equals(Axes.CHANNEL)) {
				return getColorStacks(ipds);
			}
		}
		return getMonochromeStacks(ipds);
	}

	private List<ImagePlaneDetailsStack> getMonochromeStacks(List<ImagePlaneDetails> ipds) {
		List<ImagePlaneDetailsStack> result = new ArrayList<ImagePlaneDetailsStack>();
		for (ImagePlaneDetails ipd:ipds) {
			result.add(ImagePlaneDetailsStack.makeMonochromeStack(
					ipd.coerceToMonochrome()));
		}
		return result;
	}

	protected static class ChannelGrouping implements Comparable<ChannelGrouping> {
		final public ImagePlaneDetails ipd;
		final public int z;
		final public int t;
		final public int c;
		public ChannelGrouping(ImagePlaneDetails ipd) {
			Plane omePlane = ipd.getImagePlane().getOMEPlane();
			this.ipd = ipd;
			c = omePlane.getTheC().getValue();
			t = omePlane.getTheT().getValue();
			z = omePlane.getTheZ().getValue();
		}

		public int compareTo(ChannelGrouping o) {
			int result = ipd.getImagePlane().getSeries().compareTo(
					o.ipd.getImagePlane().getSeries());
			if (result != 0) return result;
			if (z != o.z) return z - o.z;
			if (t != o.t) return t - o.t;
			return c - o.c;
		}
	}
	private List<ImagePlaneDetailsStack> getColorStacks(List<ImagePlaneDetails> ipds) {
		List<ImagePlaneDetailsStack> result = new ArrayList<ImagePlaneDetailsStack>();
		//
		// Make ChannelGrouping keys for planes that have decent OME metadata
		//
		List<ChannelGrouping> groupingPlanes = new ArrayList<ChannelGrouping>();
		for (ImagePlaneDetails ipd:ipds) {
			final ImagePlane imagePlane = ipd.getImagePlane();
			if (imagePlane.getOMEPlane() != null) {
				groupingPlanes.add(new ChannelGrouping(ipd));
			} else {
				result.add(ImagePlaneDetailsStack.makeColorStack(ipd));
			}
		}
		if (groupingPlanes.size() > 0) {
			// Order the planes so that consecutive channels of the
			// same channel-stack appear consecutively
			Collections.sort(groupingPlanes);
			ImagePlaneDetailsStack stack = new ImagePlaneDetailsStack(PlaneStack.XYCAxes);
			int lastChannel = -1;
			for (ChannelGrouping g:groupingPlanes) {
				if (g.c > lastChannel) {
					stack.add(g.ipd, 0, 0, g.c);
				} else {
					result.add(stack);
					stack = ImagePlaneDetailsStack.makeColorStack(g.ipd);
				}
				lastChannel = g.c;
			}
			result.add(stack);
		}
		return result;
	}
			
}
