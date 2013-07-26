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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.cellprofiler.imageset.filter.ImagePlaneDetails;

/**
 * @author Lee Kamentsky
 *
 * Utilities for metadata collection on behalf of CellProfiler
 */
public class MetadataUtils {
	/**
	 * This method takes a list of columns of IPDs where each column contains the IPDs
	 * for a particular channel. It finds the metadata keys which are consistent for
	 * all IPDs in which they appear for each image set (for instance, WellName is
	 * likely to be consistent, but Wavelength will differ for each column).
	 * 
	 * @param ipdList a list of columns of IPDs
	 * @return a map of metadata name to the values for each image set
	 */
	static Map<String, List<String>> getImageSetMetadata(List<List<ImagePlaneDetails>> ipdList) {
		int nrows = Integer.MAX_VALUE;
		for (List<ImagePlaneDetails> l: ipdList) {
			nrows = Math.min(nrows, l.size());
		}
		Map<String, List<String>> result = new HashMap<String, List<String>>(); 
		if (nrows == 0) return result;
		
		Set<String> badMetadata = new HashSet<String>();
		
		Map<String, String> rowMetadata = getRowMetadata(ipdList, 0, badMetadata);
		for (Map.Entry<String, String> entry:rowMetadata.entrySet()) {
			List<String> values = new ArrayList<String>(nrows);
			values.add(entry.getValue());
			result.put(entry.getKey(), values);
		}
		for (int i=1; i<nrows; i++) {
			rowMetadata = getRowMetadata(ipdList, i, badMetadata);
			for (Map.Entry<String, List<String>> entry:result.entrySet()) {
				if (rowMetadata.containsKey(entry.getKey())) {
					entry.getValue().add(rowMetadata.get(entry.getKey()));
				} else {
					badMetadata.add(entry.getKey());
				}
			}
		}
		result.keySet().removeAll(badMetadata);
		return result;
	}
	
	private static Map<String, String> getRowMetadata(List<List<ImagePlaneDetails>> ipdList, int idx, Set<String> badMetadata) { 
		Map<String, String> image_set_metadata = new HashMap<String, String>();
		for (List<ImagePlaneDetails> l: ipdList) {
			Map<String, String> metadata = l.get(idx).metadata;
			for (String key:metadata.keySet()) {
				if (! badMetadata.contains(key)) {
					String value = metadata.get(key);
					if (image_set_metadata.containsKey(key)) {
						if (! image_set_metadata.get(key).equals(value)) {
							badMetadata.add(key);
							image_set_metadata.remove(key);
						}
					} else {
						image_set_metadata.put(key, value);
					}
				}
			}
		}
		return image_set_metadata;
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
}
