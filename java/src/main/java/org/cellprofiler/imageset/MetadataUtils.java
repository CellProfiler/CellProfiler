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
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ome.xml.model.Pixels;
import ome.xml.model.Plane;
import ome.xml.model.enums.DimensionOrder;


/**
 * @author Lee Kamentsky
 *
 * Utilities for metadata collection on behalf of CellProfiler
 */
public class MetadataUtils {
	/**
	 * This method takes a list of columns of IPDs where each column contains the IPDs
	 * for a particular channel. It finds the metadata keys which are consistent for
	 * all IPDs in the image set, for instance, "Well", is likely to be consistent
	 * whereas "Wavelength" is not.
	 * 
	 * @param imageSets
	 * @param mustHave a map of metadata key to the channel index from which to extract.
	 *                 These keys are ones that we must return for each image set. 
	 * @param comparators metadata-key-specific comparators. For instance,
	 *        site metadata is usually numeric, so 1 is the same as 01, but
	 *        plate metadata is usually alphanumeric and each needs an appropriate
	 *        comparator.
	 * @return a map of metadata name to the values for each image set
	 */
	static Map<String, List<String>> getImageSetMetadata(
			List<ImageSet> imageSets,
			Map<String,Integer> mustHave,
			Map<String, Comparator<String>> comparators) {
		int nrows = imageSets.size();
		if (nrows == 0) return Collections.emptyMap();
		final List<Map<String, String>> result = new ArrayList<Map<String, String>>();
		//
		// The metadata to ignore. We populate with the must-haves
		// which we collect in a more explicit manner
		//
		final Set<String> badMetadata = new HashSet<String>();
		final Set<String> keysSeen = new HashSet<String>();
		for (ImageSet imageSet:imageSets) {
			final Map<String, String> imageSetMetadata = new HashMap<String, String>();
			result.add(imageSetMetadata);
			//
			// The "must have" entries
			//
			for (Map.Entry<String, Integer> entry:mustHave.entrySet()) {
				final ImagePlaneDetailsStack stack = imageSet.get(entry.getValue());
				for (ImagePlaneDetails ipd:stack) {
					final String key = entry.getKey();
					if (ipd.containsKey(key)) {
						imageSetMetadata.put(key, ipd.get(key));
					}
				}
			}
			//
			// Other entries that we can find
			//
			for (ImagePlaneDetailsStack stack:imageSet) {
				for (ImagePlaneDetails ipd:stack) {
					for (String key:ipd) {
						if ((! badMetadata.contains(key)) && (! mustHave.containsKey(key))) {
							String value = ipd.get(key);
							final String oldValue = imageSetMetadata.get(key); 
							if (oldValue != null) {
								if (comparators != null) {
									Comparator<String> c = comparators.get(key);
									if (c != null) {
										if (c.compare(value, oldValue) != 0) {
											badMetadata.add(key);
											continue;
										}
									} else if (! value.equals(oldValue)) {
										badMetadata.add(key);
										continue;
									}
								}
							}
							imageSetMetadata.put(key, value);
							keysSeen.add(key);
						}
					}
				}
			}
		}
		//
		// At the end, reorganize the list into columns per metadata item
		//
		keysSeen.removeAll(badMetadata);
		keysSeen.addAll(mustHave.keySet());
		final Map<String, List<String>> output = new HashMap<String, List<String>>();
		for (String key:keysSeen) {
			List<String> values = new ArrayList<String>(imageSets.size());
			for (Map<String, String>metadata:result) {
				values.add(metadata.get(key));
			}
			output.put(key, values);
		}
		return output;
	}
	
	//
	// This expression looks for:
	// two backslashes = escaped backslash
	// backslash open parentheses = escaped parentheses
	// (?P
	// (
	// The first two cases are skipped
	// The third case is a named group capture
	// The fourth case is a parentheses expression not
	// to be captured.
	final private static Pattern pythonGroupPattern = Pattern.compile(
	"(\\\\\\\\)|(\\\\\\()|(((\\(\\?P<([^>]+)>)|(\\((?!\\?P<))))");

	/**
	 * Compile a Python regular expression, converting the key extraction pattern,
	 * "(?P&lt;key&gt;...)" to a simple parentheses expression.
	 * 
	 * @param pattern the Python regular expression
	 * @param keys on input, an empty list, on output, holds the keys in the order they appear in the expression
	 * @return a java.util.regexp.Pattern that can be used for matching against the expression.
	 */
	public static Pattern compilePythonRegexp(String pattern, List<String> keys) {
		Matcher matcher = pythonGroupPattern.matcher(pattern);
		String p = "";
		int start = 0;
		while (matcher.find()) {
			if ((matcher.group(1) != null)||(matcher.group(2) != null)) {
				p += pattern.substring(start, matcher.end());
				start = matcher.end();
				continue;
			}
			p += pattern.substring(start, matcher.start()+1);
			if (keys != null) {
				if (matcher.group(6) != null) {
					keys.add(matcher.group(6));
				} else {
					keys.add(null);
				}
			}
			start = matcher.end();
		}
		p += pattern.substring(start);
		return Pattern.compile(p);
	}
	
	/**
	 * Get a plane index given the channel, z and t indices
	 * for a particular OMEXML pixels metadata node.
	 * 
	 * @param pixels an OMEXML pixels metadata node giving the layout for planes
	 * @param c the channel #
	 * @param z the z-stack height index
	 * @param t the time index
	 * @return the index to the plane.
	 */
	static public int getIndex(Pixels pixels, int c, int z, int t) {
		final DimensionOrder dimensionOrder = pixels.getDimensionOrder();
		switch(dimensionOrder) {
		case XYCTZ:
			return c + pixels.getSizeC().getValue() * (t + pixels.getSizeT().getValue()*z);
		case XYCZT:
			return c + pixels.getSizeC().getValue() * (z + pixels.getSizeZ().getValue()*t);
		case XYTCZ:
			return t + pixels.getSizeT().getValue() * (c + pixels.getSizeC().getValue() * z);
		case XYTZC:
			return t + pixels.getSizeT().getValue() * (z + pixels.getSizeZ().getValue() * c);
		case XYZCT:
			return z + pixels.getSizeZ().getValue() * (c + pixels.getSizeC().getValue() * t);
		case XYZTC:
			return z + pixels.getSizeZ().getValue() * (t + pixels.getSizeT().getValue() * c);
		}
		throw new UnsupportedOperationException(String.format("Unsupported dimension order: %s", dimensionOrder.toString()));
	}
	/**
	 * Get the plane index given an OME plane
	 * 
	 * @param plane
	 * @return
	 */
	static public int getIndex(Plane plane) {
		return getIndex(plane.getPixels(), plane.getTheC().getValue(), 
				plane.getTheZ().getValue(), plane.getTheT().getValue());	
	}
	/**
	 * Get the Z index of a plane, given its index and the
	 * OME Pixels that describe its image
	 * 
	 * @param pixels
	 * @param index
	 * @return
	 */
	static public int getZ(Pixels pixels, int index) {
		final Integer sizeC = pixels.getSizeC().getValue();
		final Integer sizeT = pixels.getSizeT().getValue();
		final Integer sizeZ = pixels.getSizeZ().getValue();
		final DimensionOrder dimensionOrder = pixels.getDimensionOrder();
		switch(dimensionOrder) {
		case XYCTZ:
		case XYTCZ:
			return index / (sizeC * sizeT);
		case XYCZT:
			return (index / sizeC) % sizeZ;
		case XYTZC:
			return (index / sizeT) % sizeZ;
		case XYZCT:
		case XYZTC:
			return index % sizeZ;
		}
		throw new UnsupportedOperationException(String.format("Unsupported dimension order: %s", dimensionOrder.toString()));
	}
	/**
	 * Get the C index of a plane, given its index and the
	 * OME Pixels that describe its image
	 * 
	 * @param pixels
	 * @param index
	 * @return
	 */
	static public int getC(Pixels pixels, int index) {
		final Integer sizeC = pixels.getSizeC().getValue();
		final Integer sizeT = pixels.getSizeT().getValue();
		final Integer sizeZ = pixels.getSizeZ().getValue();
		final DimensionOrder dimensionOrder = pixels.getDimensionOrder();
		switch(dimensionOrder) {
		case XYZTC:
		case XYTZC:
			return index / (sizeZ * sizeT);
		case XYZCT:
			return (index / sizeZ) % sizeC;
		case XYTCZ:
			return (index / sizeT) % sizeC;
		case XYCZT:
		case XYCTZ:
			return index % sizeC;
		}
		throw new UnsupportedOperationException(String.format("Unsupported dimension order: %s", dimensionOrder.toString()));
	}
	/**
	 * Get the T index of a plane, given its index and the
	 * OME Pixels that describe its image
	 * 
	 * @param pixels
	 * @param index
	 * @return
	 */
	static public int getT(Pixels pixels, int index) {
		final Integer sizeC = pixels.getSizeC().getValue();
		final Integer sizeT = pixels.getSizeT().getValue();
		final Integer sizeZ = pixels.getSizeZ().getValue();
		final DimensionOrder dimensionOrder = pixels.getDimensionOrder();
		switch(dimensionOrder) {
		case XYZCT:
		case XYCZT:
			return index / (sizeZ * sizeC);
		case XYZTC:
			return (index / sizeZ) % sizeT;
		case XYCTZ:
			return (index / sizeC) % sizeT;
		case XYTZC:
		case XYTCZ:
			return index % sizeT;
		}
		throw new UnsupportedOperationException(String.format("Unsupported dimension order: %s", dimensionOrder.toString()));
	}
}
