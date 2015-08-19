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
package org.cellprofiler.imageset.filter;

import java.util.List;

import org.cellprofiler.imageset.ImagePlaneDetails;
import org.cellprofiler.imageset.filter.Filter.BadFilterExpressionException;

/**
 * @author Lee Kamentsky
 * 
 * A predicate that parses metadata.
 *
 */
public class MetadataPredicate extends
		AbstractImagePlaneDetailsPredicate<ImagePlaneDetails> implements
		TokenParser<ImagePlaneDetails, ImagePlaneDetails> {
	final static public String SYMBOL="metadata";
	private List<FilterPredicate<ImagePlaneDetails, ?>> subpredicates;
	
	static class MetadataKeyPredicate extends AbstractImagePlaneDetailsPredicate<String> {
		private final String key;
		private String value;
		
		public MetadataKeyPredicate(String key) {
			this.key = key;
		}
		/* (non-Javadoc)
		 * @see org.cellprofiler.imageset.filter.FilterPredicate#getSymbol()
		 */
		public String getSymbol() {
			return key;
		}

		/* (non-Javadoc)
		 * @see org.cellprofiler.imageset.filter.FilterPredicate#setSubpredicates(org.cellprofiler.imageset.filter.FilterPredicate<TOUT,?>[])
		 */
		public void setSubpredicates(List<FilterPredicate<String, ?>> subpredicates) throws BadFilterExpressionException {
			throw new AssertionError("Metadata key predicates have literals, not subpredicates.");
		}

		/* (non-Javadoc)
		 * @see org.cellprofiler.imageset.filter.FilterPredicate#setLiteral(java.lang.String)
		 */
		public void setLiteral(String literal) throws BadFilterExpressionException {
			value = literal;
		}

		/* (non-Javadoc)
		 * @see org.cellprofiler.imageset.filter.FilterPredicate#eval(java.lang.Object)
		 */
		public boolean eval(ImagePlaneDetails candidate) {
			String value = candidate.get(key);
			return (value != null && this.value.equals(value));
		}

		/* (non-Javadoc)
		 * @see org.cellprofiler.imageset.filter.FilterPredicate#getOutputClass()
		 */
		public Class<String> getOutputClass() {
			return null;
		}
	}
	/**
	 * The MetadataDoesPredicate parses the token that follows it, which is the metadata key.
	 * Otherwise, it behaves just like DoesPredicate
	 * 
	 * @author Lee Kamentsky
	 *
	 */
	static class MetadataDoesPredicate 
	extends DoesPredicate<ImagePlaneDetails> 
	implements TokenParser<ImagePlaneDetails, ImagePlaneDetails> {

		public MetadataDoesPredicate() {
			super(ImagePlaneDetails.class);
		}

		/* (non-Javadoc)
		 * @see org.cellprofiler.imageset.filter.TokenParser#parse(java.lang.String)
		 */
		public FilterPredicate<ImagePlaneDetails, ?> parse(String token) {
			return new MetadataKeyPredicate(token);
		}
	}
	/**
	 * The MetadataDoesNotPredicate parses the token that follows it, which is the metadata key.
	 * Otherwise, it behaves just like DoesNotPredicate
	 * 
	 * @author Lee Kamentsky
	 *
	 */
	static class MetadataDoesNotPredicate 
	extends DoesNotPredicate<ImagePlaneDetails> 
	implements TokenParser<ImagePlaneDetails, ImagePlaneDetails> {

		public MetadataDoesNotPredicate() {
			super(ImagePlaneDetails.class);
		}

		/* (non-Javadoc)
		 * @see org.cellprofiler.imageset.filter.TokenParser#parse(java.lang.String)
		 */
		public FilterPredicate<ImagePlaneDetails, ?> parse(String token) {
			return new MetadataKeyPredicate(token);
		}
	}
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getSymbol()
	 */
	public String getSymbol() {
		return SYMBOL;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#setSubpredicates(org.cellprofiler.imageset.filter.FilterPredicate<TOUT,?>[])
	 */
	public void setSubpredicates(List<FilterPredicate<ImagePlaneDetails, ?>> subpredicates) throws BadFilterExpressionException {
		this.subpredicates = subpredicates;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#setLiteral(java.lang.String)
	 */
	public void setLiteral(String literal) throws BadFilterExpressionException {
		throw new AssertionError("The metadata predicate does not accept a literal argument");
		
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#eval(java.lang.Object)
	 */
	public boolean eval(ImagePlaneDetails candidate) {
		return subpredicates.get(0).eval(candidate);
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.FilterPredicate#getOutputClass()
	 */
	public Class<ImagePlaneDetails> getOutputClass() {
		return ImagePlaneDetails.class;
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.filter.TokenParser#parse(java.lang.String)
	 */
	public FilterPredicate<ImagePlaneDetails, ?> parse(String token) {
		if (token.equals(MetadataDoesPredicate.SYMBOL)) {
			return new MetadataDoesPredicate();
		} else if (token.equals(MetadataDoesNotPredicate.SYMBOL)) {
			return new MetadataDoesNotPredicate();
		}
		throw new AssertionError("Do or do not, there is no try.");
	}
}
