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
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.logging.Logger;



/**
 * @author Lee Kamentsky
 *
 * A joiner joins two lists of IPD stacks together
 * using a set of keys
 *
 */
public class Joiner {
	protected static class Join {
		final public ChannelFilter channelFilter;
		final public List<MetadataKeyPair> joiningKeys;
		public Join(ChannelFilter channelFilter, List<MetadataKeyPair> joiningKeys) {
			this.channelFilter = channelFilter;
			this.joiningKeys = joiningKeys;
		}
		public List<String> getLeftKey(ImagePlaneDetailsStack d) {
			List<String> result = new ArrayList<String>(joiningKeys.size());
			for (MetadataKeyPair joiningKey:joiningKeys) {
				result.add(d.get(joiningKey.leftKey));
			}
			return result;
		}
		public List<String> getRightKey(ImagePlaneDetailsStack d) {
			List<String> result = new ArrayList<String>(joiningKeys.size());
			for (MetadataKeyPair joiningKey:joiningKeys) {
				result.add(d.get(joiningKey.rightKey));
			}
			return result;
		}
		public Comparator<List<String>> getComparator() {
			List<Comparator<String>> comparators = new ArrayList<Comparator<String>>();
			for (MetadataKeyPair kp:joiningKeys) comparators.add(kp.comparator);
			return new ListComparator<String>(comparators);
		}
	}
	final private Comparator<List<String>> imageSetKeyComparator;
	final private static Logger logger = Logger.getLogger(Joiner.class.getCanonicalName());
	final private ChannelFilter anchorChannel;
	final private List<String> keys;
	final private List<Join> joinedChannels = new ArrayList<Joiner.Join>();
	/**
	 * Construct a joiner to join other channels to an anchor
	 * channel by metadata
	 * 
	 * @param anchorChannel the anchor channel produces one stack
	 *                      per image set and all other channels
	 *                      join to this one.
	 * @param keys the keys that uniquely define an image set in
	 *             the anchor channel's stack metadata (in lexical
	 *             order for channel sorting).
	 *             
	 * @param comparators - comparators for each of the keys
	 */
	public Joiner(ChannelFilter anchorChannel, List<String> keys, List<Comparator<String>> comparators){
		this.anchorChannel = anchorChannel;
		this.keys = keys;
		imageSetKeyComparator = new ListComparator<String>(comparators);
	}
	
	/**
	 * Add another channel to the joiner as well as a join criterion
	 * 
	 * @param channelFilter
	 * @param joiningKeys - keys to use to join stacks from this channel to those in the anchor channel.
	 */
	public void addChannel(ChannelFilter channelFilter, List<MetadataKeyPair> joiningKeys) {
		joinedChannels.add(new Join(channelFilter, joiningKeys));
	}
	
	/**
	 * Make image sets using the joiner
	 * 
	 * @param ipds the image planes to assemble into image sets
	 * @param errors an empty list which will hold the errors that we found
	 * 
	 * @return an ordered list of image sets
	 */
	public List<ImageSet> join(List<ImagePlaneDetails> ipds, List<ImageSetError> errors)
	{
		SortedMap<List<String>, ImageSet> imageSets = 
			new TreeMap<List<String>, ImageSet>(imageSetKeyComparator);
		SortedMap<List<String>, List<ImagePlaneDetailsStack>> duplicates =
			new TreeMap<List<String>, List<ImagePlaneDetailsStack>>(imageSetKeyComparator);
		List<ImageSetMissingError> missing = new ArrayList<ImageSetMissingError>();
		Set<List<String>> bad = new TreeSet<List<String>>(imageSetKeyComparator);
		
		stack_loop:
		for (ImagePlaneDetailsStack stack:anchorChannel.makeStacks(ipds)) {
			List<String> stackKey = new ArrayList<String>(keys.size());
			for (String key:keys) {
				final String value = stack.get(key);
				if (value == null) {
					ImageFile f = stack.iterator().next().getImagePlane().getImageFile();
					logger.info(String.format("%s in channel %s does not have metadata for %s key", 
							f, anchorChannel.getName(), key));
					continue stack_loop;
				}
				stackKey.add(value);
			}
			if (imageSets.containsKey(stackKey)) {
				if (! duplicates.containsKey(stackKey)) {
					duplicates.put(stackKey, new ArrayList<ImagePlaneDetailsStack>(1));
				}
				duplicates.get(stackKey).add(stack);
			} else {
				final ImageSet imageSet = new ImageSet(stack, stackKey);
				imageSets.put(stackKey, imageSet);
			}
		}
		errors.addAll(compileDuplicates(duplicates, anchorChannel.getName()));
		for (int i=0; i<joinedChannels.size(); i++) {
			Join join = joinedChannels.get(i);
			SortedMap<List<String>, List<ImageSet>> groupedImageSets =
				new TreeMap<List<String>, List<ImageSet>>(join.getComparator());
			duplicates = new TreeMap<List<String>, List<ImagePlaneDetailsStack>>(join.getComparator());
			//
			// Create groups of extant image sets using the possibly truncated
			// key set
			//
			for (ImageSet imageSet:imageSets.values()) {
				final List<String> stackKey = join.getLeftKey(imageSet.get(0));
				List<ImageSet> match = groupedImageSets.get(stackKey);
				if (match == null) {
					match = new ArrayList<ImageSet>();
					groupedImageSets.put(stackKey, match);
				}
				match.add(imageSet);
			}
			//
			// Generate the auxilliary channel's stacks and assign
			//
			for (ImagePlaneDetailsStack stack:join.channelFilter.makeStacks(ipds)) {
				final List<String> stackKey = join.getRightKey(stack);
				final List<ImageSet> match = groupedImageSets.get(stackKey);
				if (match == null) {
					ImageFile f = stack.iterator().next().getImagePlane().getImageFile();
					String message = String.format(
						"%s in channel %s with metadata %s has no match", 
						f, anchorChannel.getName(), stackKey);
					missing.add(new ImageSetMissingError(
							join.channelFilter.getName(),
							message, stackKey));
				} else {
					for (ImageSet imageSet:match) {
						if (imageSet.size() > i+1) {
							List<ImagePlaneDetailsStack> dlist = duplicates.get(imageSet.getKey());
							if (dlist == null) {
								dlist = new ArrayList<ImagePlaneDetailsStack>();
								duplicates.put(imageSet.getKey(), dlist);
							}
							dlist.add(stack);
						} else {
							imageSet.add(stack);
						}
					}
				}
			}
			errors.addAll(compileDuplicates(duplicates, join.channelFilter.getName()));
			for (ImageSet imageSet:imageSets.values()) {
				if (imageSet.size() != i+2) {
					missing.add(new ImageSetMissingError(join.channelFilter.getName(), "Missing from channel", imageSet.getKey()));
					imageSet.add(null);
					bad.add(imageSet.getKey());
				}
			}
		}
		//
		// At the end, remove all bad image sets
		//
		for (List<String> key:bad) imageSets.remove(key);
		errors.addAll(missing);
		return new ArrayList<ImageSet>(imageSets.values());
	}

	/**
	 * Create ImageSetDuplicateError entries for these duplicates
	 *  
	 * @param duplicates stacks that are duplicates for this channel.
	 * @param name channel name
	 * @return
	 */
	private List<? extends ImageSetError> compileDuplicates(
			SortedMap<List<String>, List<ImagePlaneDetailsStack>> duplicates,
			String name) {
		List<ImageSetError> result = new ArrayList<ImageSetError>();
		for (Map.Entry<List<String>, List<ImagePlaneDetailsStack>> e:duplicates.entrySet()) {
			result.add(new ImageSetDuplicateError(name, "Duplicate entries", e.getKey(), e.getValue()));
		}
		return result;
	}

}
