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

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

/**
 * @author Lee Kamentsky
 *
 * The Details class represents a collection of metadata
 * values. There are fewer operations exposed than Map<K,V>
 * and values can only be added, not removed. Each instance
 * has an optional parent Details which supplies metadata
 * values from the containing hierarchy/
 */
public class Details implements Iterable<String> {
	final private Map<String, String> metadata = new HashMap<String, String>();
	protected final Details parent;
	
	public Details() {
		parent = null;
	}
	
	public Details(Details parent) {
		this.parent = parent;
	}
	
	/**
	 * Make a copy of the child.
	 * @param parent
	 * @param child
	 */
	protected Details(Details parent, Details child) {
		this(parent);
		this.metadata.putAll(child.metadata);
	}
	
	public boolean containsKey(String key) {
		return metadata.containsKey(key) || ((parent != null) && (parent.containsKey(key)));
	}
	
	public String get(String key) {
		String result = metadata.get(key);
		return ((result == null) && (parent != null))? parent.get(key) : result;
	}
	
	public void put(String key, String value) {
		metadata.put(key, value);
	}
	
	/**
	 * Add all metadata items in the map to our metadata
	 * 
	 * @param map
	 */
	public void putAll(Map<String, String> map) {
		metadata.putAll(map);
	}
	
	/**
	 * @return an iterator over metadata keys.
	 */
	public Iterator<String> iterator() {
		if (parent == null) return metadata.keySet().iterator();
		return new Iterator<String> () {
			Iterator<String> metadataIterator = metadata.keySet().iterator();
			Iterator<String> parentIterator = parent.iterator();
			String next = null;
			public boolean hasNext() {
				if (next != null) return true;
				if (metadataIterator.hasNext()) {
					next = metadataIterator.next();
					return true;
				}
				while (parentIterator.hasNext()) {
					final String nextCandidate = parentIterator.next();
					if (! metadata.containsKey(nextCandidate)) {
						next = nextCandidate;
						return true;
					}
				}
				next = null;
				return false;
			}
			public String next() {
				if (! hasNext()) {
					throw new NoSuchElementException();
				}
				String retval = next;
				next = null;
				return retval;
			}
			public void remove() {
				// TODO Auto-generated method stub
				
			}
		};
	}

}
