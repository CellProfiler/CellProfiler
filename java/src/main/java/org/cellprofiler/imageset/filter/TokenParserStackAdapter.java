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

package org.cellprofiler.imageset.filter;

import org.cellprofiler.imageset.ImagePlaneDetails;
import org.cellprofiler.imageset.ImagePlaneDetailsStack;

/**
 * @author Lee Kamentsky
 *
 */
public class TokenParserStackAdapter<T> extends StackAdapter<T> implements
		TokenParser<ImagePlaneDetailsStack, T> {
	TokenParser<ImagePlaneDetails, T> tp;
	static public <T> StackAdapter<T> makeAdapter(TokenParser<ImagePlaneDetails, T> p) {
		Class<T> klass = p.getOutputClass();
		TokenParserStackAdapter<T> adapter = new TokenParserStackAdapter<T>(klass);
		adapter.planeFilterPredicate = p;
		adapter.tp = p;
		return adapter;
	}
	protected TokenParserStackAdapter(Class<T> klass) {
		super(klass);
	}
	public FilterPredicate<T, ?> parse(String token) {
		return tp.parse(token);
	}

}
