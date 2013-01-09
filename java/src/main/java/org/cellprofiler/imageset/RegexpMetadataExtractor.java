/**
 * 
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
	final private static Pattern pythonGroupPattern = Pattern.compile(
			"(?<!\\\\)(?<=\\()\\?P<([^>]+)>");
	
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
		Matcher matcher = pythonGroupPattern.matcher(pattern);
		String p = "";
		int start = 0;
		List<String> keys = new ArrayList<String>();
		while (matcher.find()) {
			p += pattern.substring(start, matcher.start());
			keys.add(matcher.group(1));
			start = matcher.end();
		}
		p += pattern.substring(start);
		this.pattern = Pattern.compile(p);
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
