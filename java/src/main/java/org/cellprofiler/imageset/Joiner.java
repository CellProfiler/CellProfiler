/**
 * CellProfiler is distributed under the GNU General Public License.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2013 Broad Institute
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

import org.cellprofiler.imageset.filter.Filter;
import org.cellprofiler.imageset.filter.ImagePlaneDetails;

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
		final private Filter filter;
		
		final private String name;
		/**
		 * Initialize the channel filter with a filter and metadata keys.
		 * 
		 * @param channelName
		 * @param keys
		 * @param filter
		 */
		public ChannelFilter(String channelName, String [] keys, Filter filter) {
			this.name = channelName;
			this.keys = keys;
			this.filter = filter;
		}
		
		/**
		 * Initialize the channel filter to match by order.
		 * 
		 * @param channelName
		 * @param filter
		 */
		public ChannelFilter(String channelName, Filter filter) {
			this(channelName, null, filter);
		}
		
		public Filter getFilter() { return filter; }
		
		public String getName() { return name; }
		
		public String [] getKeys() { return keys; }
		
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
	public Joiner(List<ChannelFilter> channels) throws JoinerException {
		if (channels.size() == 0) {
			throw new JoinerException("The image set must have at least one channel");
		}
		this.channels = new ArrayList<ChannelFilter>(channels);
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
	 * Join the image plane descriptors to make an image set list
	 * @param ipds image plane descriptors (= image planes + metadata)
	 * @param errors on input, this should be an empty collection. 
	 *        On output it will hold an ImageSetError for every image set key
	 *        which had missing or duplicate images.
	 * @return a list of image sets.
	 */
	public <T  extends ImagePlaneDetails> List<ImageSet> join(List<T> ipds, Collection<ImageSetError> errors) {
		Map<List<String>, List<ImagePlaneDetails>> result = 
			new HashMap<List<String>, List<ImagePlaneDetails>>();
		/*
		 * Do the anchor channel first.
		 */
		Map<List<String>, ImagePlaneDetails> channelResult = filterChannel(
				channels.get(anchorChannel), ipds, errors);
		for (Map.Entry<List<String>, ImagePlaneDetails> entry:channelResult.entrySet()) {
			List<ImagePlaneDetails> imagesetIPDs = getListOfNulls(channels.size());
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
			for (Map.Entry<List<String>, List<ImagePlaneDetails>> entry:result.entrySet()) {
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
				final List<ImagePlaneDetails> errantImageSet = new ArrayList<ImagePlaneDetails>(channels.size());
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
				List<ImagePlaneDetails> imageSet = result.get(key);
				error.setImageSet(imageSet);
				result.remove(key);
			}
		}
		/*
		 * At the end, sort into a list by alphabetical metadata tag value
		 */
		List<List<String>> keysInOrder = new ArrayList<List<String>>(result.keySet());
		Collections.sort(keysInOrder, new Comparator<List<String>>() {

			public int compare(List<String> o1, List<String> o2) {
				for (int i=0; i<Math.min(o1.size(), o2.size()); i++) {
					int result = o1.get(i).compareTo(o2.get(i));
					if (result != 0) return result;
				}
				return o1.size() - o2.size();
			}
		});
		/*
		 * Construct the final list
		 */
		List<ImageSet> imageSet = new ArrayList<ImageSet>(keysInOrder.size());
		for (List<String> key:keysInOrder) {
			imageSet.add(new ImageSet(result.get(key), key));
		}
		return imageSet;
	}
	
	private <T extends ImagePlaneDetails> Map<List<String>, ImagePlaneDetails> filterChannel(
			ChannelFilter channelFilter,
			List<T> ipds,
			Collection<ImageSetError> errors) {
		Map<List<String>, ImageSetDuplicateError> localErrors = 
			new HashMap<List<String>, ImageSetDuplicateError>();
		Map<List<String>, ImagePlaneDetails> result = new HashMap<List<String>, ImagePlaneDetails>();
		String [] keys = channelFilter.getKeys();
		int [] idxs = channelFilter.getJoiningKeyIndices();
		next_ipd:
		for (ImagePlaneDetails ipd:ipds) {
			if (channelFilter.getFilter().eval(ipd)) {
				List<String> keyValues = new ArrayList<String>(idxs.length);
				for (int i:idxs) {
					if (! ipd.metadata.containsKey(keys[i])) continue next_ipd;
					keyValues.add(ipd.metadata.get(keys[i]));
				}
				if (result.containsKey(keyValues)) {
					if (localErrors.containsKey(keyValues)) {
						localErrors.get(keyValues).getImagePlaneDetails().add(ipd);
					} else {
						List<ImagePlaneDetails> errorIPDs = new ArrayList<ImagePlaneDetails>(2);
						errorIPDs.add(result.get(keyValues));
						errorIPDs.add(ipd);
						localErrors.put(keyValues, new ImageSetDuplicateError(
								channelFilter.getName(), "Duplicate entries for key", keyValues, errorIPDs));
					}
					continue;
				}
				result.put(keyValues, ipd);
			}
		}
		errors.addAll(localErrors.values());
		return result;
	}
			
}
