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
	final private List<String> metadataKeys;
	
	/**
	 * Constructor initializes the extractor with the regular expression
	 * that's used to extract metadata from a string.
	 * 
	 * @param pattern a Python regular expression that uses (?P&lt;key&gt;...) to
	 *       extract metadata from the string.
	 */
	public RegexpMetadataExtractor(String pattern) {
		List<String> keys = new ArrayList<String>();
		List<String> metadataKeys = new ArrayList<String>();
		this.pattern = MetadataUtils.compilePythonRegexp(pattern, keys);
		this.keys = new String [keys.size()];
		for (int i=0; i<keys.size(); i++) {
			final String key = keys.get(i);
			if (key == null) {
				this.keys[i] = null;
			} else {
				this.keys[i] = StringCache.intern(key);
				metadataKeys.add(key);
			}
		}
		this.metadataKeys = Collections.unmodifiableList(metadataKeys);
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
				if ((value != null) && (keys[i] != null)) {
					map.put(keys[i], StringCache.intern(value));
				}
			}
		}
		return map;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractor#getMetadataKeys()
	 */
	public List<String> getMetadataKeys() {
		return metadataKeys;
	}

}
