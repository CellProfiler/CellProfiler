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
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author Lee Kamentsky
 *
 * Regular expression metadata extractor
 */
public class RegexpMetadataExtractor implements MetadataExtractor<String> {
	/*
	 * The Python regexp parser captures metadata using named groups
	 * of the form, "(?P<foo>...)". This pattern captures the name and
	 * makes it possible to convert from Python to Java.
	 */
	final private Pattern pattern;
	final private String [] keys;
	
	/**
	 * Constructor initializes the extractor with the regular expression
	 * that's used to extract metadata from a string.
	 * 
	 * @param pattern a Python regular expression that uses (?P&lt;key&gt;...) to
	 *       extract metadata from the string.
	 */
	public RegexpMetadataExtractor(String pattern) {
		List<String> keys = new ArrayList<String>();
		this.pattern = MetadataUtils.compilePythonRegexp(pattern, keys);
		this.keys = keys.toArray(new String [keys.size()]);
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractor#extract(java.lang.Object)
	 */
	public Map<String, String> extract(String source) {
		HashMap<String, String> map = new HashMap<String, String>();
		Matcher matcher = pattern.matcher(source);
		if (matcher.find()) {
			for (int i=0; i<matcher.groupCount(); i++) {
				String value = matcher.group(i+1);
				if (value != null) {
					map.put(keys[i], value);
				}
			}
		}
		return map;
	}

}
