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

import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.cellprofiler.imageset.ImageFile;
import org.cellprofiler.imageset.ImagePlane;
import org.cellprofiler.imageset.MetadataExtractor;

/**
 * @author Lee Kamentsky
 *
 * A filter parses a filter expression and evaluates candidate
 * ImagePlaneDetails on the basis of the filter.
 * 
 */
public class Filter {
	public static class BadFilterExpressionException extends Exception {

		private static final long serialVersionUID = -3134183711783453007L;
		public int offset;
		public String expression;
		private String reason;
		/**
		 * Construct an exception reporting on a bad parse of a filter expression
		 * @param message message to user
		 * @param expression expression at fault
		 * @param offset offset to likely location of error
		 */
		public BadFilterExpressionException(String message, String expression, int offset) {
			super(String.format("%s: %s->%s", message, 
					expression.substring(0, offset), expression.substring(offset)));
			this.offset = offset;
			this.expression = expression;
			this.reason = message;
		}
		
		/**
		 * Construct an exception reporting a bad parse at an unknown location
		 * @param message
		 */
		public BadFilterExpressionException(String message) {
			super(message);
		}
		
		/**
		 * Throw a BadFilterExpressionException from a parser outside of the one that
		 * did the throwing.
		 * 
		 * @param expression the expression containing the sub-expression
		 * @param offset the offset of the subexpression that did the throwing
		 * @throws BadFilterExpressionException
		 */
		public void rethrow(String expression, int offset) throws BadFilterExpressionException{
			throw new BadFilterExpressionException(reason, expression, this.offset + offset);
		}
	}
	static private int nCachedEntries = 100;
	final static private Map<String, Filter> filterCache = new HashMap<String, Filter>();
	final static public Random random = new Random(0);
	final public FilterPredicate<ImagePlaneDetails, ?> rootPredicate;
	final private String expression;
	/**
	 * @return maximum number of filters to be cached
	 */
	static public int getCachedEntryCount() {
		return nCachedEntries;
	}
	/**
	 * Set the maximum number of filters to be cached
	 * 
	 * @param n number of filters to be cached
	 */
	static public void setCachedEntryCount(int n) {
		nCachedEntries = n;
	}
	/**
	 * Construct a filter from an expression
	 * 
	 * @param expression a filter expression
	 * @throws BadFilterExpressionException if there was an error parsing the expression
	 */
	public Filter(String expression) throws BadFilterExpressionException {
		this.expression = expression;
		String [] rest = new String [1];
		rootPredicate = parse(expression, ImagePlaneDetails.class, rest);
	}
	
	/**
	 * Filter an IPD based on a filter expression
	 * 
	 * @param expression construct a filter from this expression (or use
	 *                   the cached filter).
	 * @param ipd image plane + details to be filtered
	 * 
	 * @return true to keep, false to filter out
	 * @throws BadFilterExpressionException if the filter syntax is badly formed
	 */
	static public boolean filter(String expression, ImagePlaneDetails ipd) 
		throws BadFilterExpressionException {
		return getFilter(expression).eval(ipd);
	}
	
	/**
	 * Filter a URL based on a filter expression
	 * 
	 * @param expression expression to use to evaluate the URL
	 * @param url construct an ImageFile based on this URL
	 * @return true to keep, false to filter out
	 * @throws BadFilterExpressionException if the expression cannot be parsed
	 */
	static public boolean filter(String expression, URI url) throws BadFilterExpressionException {
		return filter(expression, new ImagePlaneDetails(
				new ImagePlane(new ImageFile(url)), MetadataExtractor.emptyMap));
	}
	/**
	 * Filter a URL based on a filter expression
	 * 
	 * @param expression expression to use to evaluate the URL
	 * @param url URL as a string
	 * @return true if passes filter
	 * @throws BadFilterExpressionException if filter was not in correct format
	 * @throws URISyntaxException 
	 */
	static public boolean filter(String expression, String url) 
	throws BadFilterExpressionException, URISyntaxException {
		return filter(expression, new URI(url));
	}
	
	static private Filter getFilter(String expression) throws BadFilterExpressionException {
		synchronized(filterCache) {
			if (! filterCache.containsKey(expression)) {
				if (filterCache.size() >= nCachedEntries) {
					/*
					 * Decimate the cache by half.
					 */
					int halfSize = filterCache.size() / 2;
					List<String> expressions = new ArrayList<String>(filterCache.keySet());
					for (int i=0; i<halfSize; i++) {
						int idx = random.nextInt(expressions.size());
						filterCache.remove(expressions.get(idx));
						expressions.remove(idx);
					}
				}
				final Filter filter = new Filter(expression);
				filterCache.put(expression, filter);
				return filter;
			} else {
				return filterCache.get(expression);
			}
		}
	}
	/**
	 * Determine whether a particular image plane passes the filter
	 * @param ipd image plane + metadata
	 * @return true if the image plane passes the filter, false if it is filtered out
	 */
	public boolean eval(ImagePlaneDetails ipd) {
		return rootPredicate.eval(ipd);
	}
	/**
     * (?:\\.|[^ )]) matches either backslash-anything or anything but
     * space and parentheses. So you can have variable names with spaces
     * and that's needed for arbitrary metadata names
	 */
	static private Pattern tokenPattern = Pattern.compile("((?:\\\\.|[^ )])+) ?");
	/**
	 * A literal is a quote followed by a quote-escape encoded string. The pattern
	 * consumes a single trailing space.
	 * 
	 * "[^\\\\\"]" = [^\\"] matches anything except for backslash and quote
	 * "(?:\\\\.)" = (?:\.) matches backslash-anything without capturing it
	 * (?:[^\\\\\"]|(?:\\\\.)) matches a single escape-encoded character without
	 * capturing it.
	 */
	static private Pattern literalPattern = Pattern.compile("\"((?:[^\\\\\"]|(?:\\\\.))*)\" ?");
	/**
	 * Find backslash-escaped characters in a quote-escaped string. 
	 */
	static private Pattern quoteEscapePattern = Pattern.compile("\\\\(.)");
	/*
	 * Parentheses expressions are separated by a space. A series of parentheses
	 * expressions is terminated either by end of expression or the end parenthesis
	 * of an enclosing parenthetical expression (e.g. "((foo bar) (foo baz))")
	 *                                                                    ^
	 */
	static private Pattern endParenthesesPattern = Pattern.compile("\\)( |(?=\\))|$)");
	/**
	 * Cast a filter predicate of indeterminate input type to a specific type with
	 * a check for compatibility.
	 * 
	 * @param <T> the desired input type
	 * @param p a filter predicate
	 * @param klass the class of the desired input type
	 * @return the cast predicate
	 * @throws BadFilterExpressionException
	 */
	@SuppressWarnings("unchecked")
	static private <T> FilterPredicate<T, ?>cast(FilterPredicate<?, ?> p, Class<T> klass)
	throws BadFilterExpressionException
	{
		if (! p.getInputClass().isAssignableFrom(klass)) {
			throw new BadFilterExpressionException(String.format(
					"The %s predicate expects a %s as an input, not %s",
					p.getSymbol(), p.getInputClass().getName(), klass.getName()));
		}
		return (FilterPredicate<T, ?>) p;
	}
	/**
	 * Get the filter predicate for a given key
	 * @param <T> the input type of the filter predicate
	 * @param key the key identifying the filter predicate
	 * @param klass the class of the input type of the filter predicate
	 * @return the filter predicate
	 */
	static private <T> FilterPredicate<T, ?> get(String key, Class<T> klass) throws BadFilterExpressionException {
		if (key.equals(AndPredicate.SYMBOL)) return new AndPredicate<T>(klass);
		if (key.equals(OrPredicate.SYMBOL)) return new OrPredicate<T>(klass);
		if (key.equals(DoesPredicate.SYMBOL)) return new DoesPredicate<T>(klass);
		if (key.equals(DoesNotPredicate.SYMBOL)) return new DoesNotPredicate<T>(klass);
		if (klass.isAssignableFrom(ImagePlaneDetails.class)) {
			if (key.equals(FileNamePredicate.SYMBOL)) return cast(new FileNamePredicate(), klass);
			if (key.equals(PathPredicate.SYMBOL)) return cast(new PathPredicate(), klass);
			if (key.equals(ExtensionPredicate.SYMBOL)) return cast (new ExtensionPredicate(), klass);
			if (key.equals(MetadataPredicate.SYMBOL)) return cast(new MetadataPredicate(), klass);
			if (key.equals(ImagePredicate.SYMBOL)) return cast(new ImagePredicate(), klass);
			if (key.equals(IsColorPredicate.SYMBOL)) return cast(new IsColorPredicate(), klass);
			if (key.equals(IsMonochromePredicate.SYMBOL)) return cast(new IsMonochromePredicate(), klass);
			if (key.equals(IsStackPredicate.SYMBOL)) return cast(new IsStackPredicate(), klass);
			if (key.equals(IsStackFramePredicate.SYMBOL)) return cast(new IsStackFramePredicate(), klass);
		}
		if (klass.isAssignableFrom(String.class)) {
			if (key.equals(ContainsPredicate.SYMBOL)) return cast(new ContainsPredicate(), klass);
			if (key.equals(ContainsRegexpPredicate.SYMBOL)) return cast(new ContainsRegexpPredicate(), klass);
			if (key.equals(EndsWithPredicate.SYMBOL)) return cast(new EndsWithPredicate(), klass);
			if (key.equals(EqualsPredicate.SYMBOL)) return cast(new EqualsPredicate(), klass);
			if (key.equals(StartsWithPredicate.SYMBOL)) return cast(new StartsWithPredicate(), klass);
			if (key.equals(IsFlexPredicate.SYMBOL)) return cast(new IsFlexPredicate(), klass);
			if (key.equals(IsImagePredicate.SYMBOL)) return cast(new IsImagePredicate(), klass);
			if (key.equals(IsJPEGPredicate.SYMBOL)) return cast(new IsJPEGPredicate(), klass);
			if (key.equals(IsMoviePredicate.SYMBOL)) return cast(new IsMoviePredicate(), klass);
			if (key.equals(IsPNGPredicate.SYMBOL)) return cast(new IsPNGPredicate(), klass);
			if (key.equals(IsTifPredicate.SYMBOL)) return cast(new IsTifPredicate(), klass);
		}
		throw new BadFilterExpressionException(String.format("No applicable predicate for token %s", key));
	}
	/**
	 * Parse an expression into filter predicates
	 * 
	 * @param <T> the input type to the root filter predicate
	 * 
	 * @param expression the string expression to be parsed
	 * @param klass the acceptable class for the filter predicate
	 * @param rest upon entry, an array of strings of length 1, upon exit, the unparsed portion of the expression
	 * @return the parsed root filter predicate.
	 */
	static private <T> FilterPredicate<T, ?> parse(String expression, Class<T> klass, String [] rest)
	throws BadFilterExpressionException	{
		return parse(expression, klass, rest, null);
	}
	
	/**
	 * @param <T> the input type to the root filter predicate
	 * @param expression the string expression to be parsed
	 * @param klass the acceptable class for the filter predicate
	 * @param rest upon entry, an array of strings of length 1, upon exit, the unparsed portion of the expression
	 * @param tokenParser if not null, a token parser that produces a filter predicate for the first token.
	 * @return
	 * @throws BadFilterExpressionException
	 */
	static private <T> FilterPredicate<T, ?> parse(
			String expression, 
			Class<T> klass, 
			String [] rest, 
			TokenParser<?, T> tokenParser) 
			throws BadFilterExpressionException {
		final String token = getToken(expression, rest);
		final FilterPredicate<T, ?> p = (tokenParser == null)?get(token, klass):tokenParser.parse(token);
		if (rest[0].length() > 0) {
			final Matcher literalMatch = literalPattern.matcher(rest[0]);
			if (literalMatch.lookingAt()) {
				final String literal = literalMatch.group(1);
				String decodedLiteral = escapeDecode(literal);
				p.setLiteral(decodedLiteral);
				rest[0] = rest[0].substring(literalMatch.end());
				if (rest[0].length() == 0) return p;
			}
			if (rest[0].charAt(0) == ')') {
				return p;
			}
			if (rest[0].charAt(0) == '(') {
				parseSubpredicates(rest[0], rest, p);
			} else {
				parseSubpredicate(rest[0], rest, p);
			}
		}
		return p;
	}
	/**
	 * Escape decode (remove first backslash of backslash + character)
	 * a string. Note that this does not convert \n to LF, etc.
	 * 
	 * @param literal input string
	 * @return escape decoded string
	 */
	private static String escapeDecode(final String literal) {
		final Matcher quoteEscapeMatch = quoteEscapePattern.matcher(literal);
		String decodedLiteral = "";
		int start = 0;
		while(quoteEscapeMatch.find()) {
			final int qeStart = quoteEscapeMatch.start();
			final String qeChar = quoteEscapeMatch.group(1);
			decodedLiteral += literal.substring(start, qeStart) + qeChar;
			start = quoteEscapeMatch.end();
		}
		decodedLiteral += literal.substring(start);
		return decodedLiteral;
	}
	
	static String getToken(String expression, String [] rest) throws BadFilterExpressionException{
		final Matcher match = tokenPattern.matcher(expression);
		if (! match.lookingAt()) {
			throw new BadFilterExpressionException("Failed to parse the next token");
		}
		final String token = escapeDecode(match.group(1));
		rest[0] = expression.substring(match.end());
		return token;
	}
	static private <T> void parseSubpredicates(
			String expression, String [] rest, FilterPredicate<?, T> p) 
	throws BadFilterExpressionException {
		List<FilterPredicate<T, ?>> subpredicates = new ArrayList<FilterPredicate<T, ?>>();
		rest[0] = expression;
		while ((rest[0].length() > 0) && (rest[0].charAt(0) == '(')) {
			final String subexpression = rest[0].substring(1);
			if (p instanceof TokenParser) {
				TokenParser<?, T> tp = (TokenParser<?, T>) p; 
				subpredicates.add(parse(subexpression, p.getOutputClass(), rest, tp));
			} else {
				subpredicates.add(parse(subexpression, p.getOutputClass(), rest));
			}
			Matcher endParenthesesMatch = endParenthesesPattern.matcher(rest[0]);
			if (! endParenthesesMatch.lookingAt()) {
				throw new BadFilterExpressionException("Missing end parenthesis");
			}
			rest[0] = rest[0].substring(endParenthesesMatch.end());
		}
		p.setSubpredicates(subpredicates);
	}
	static private <T> void parseSubpredicate(String expression, String [] rest, FilterPredicate<?, T> p) 
	throws BadFilterExpressionException {
		List<FilterPredicate<T, ?>> subpredicates = new ArrayList<FilterPredicate<T, ?>>();
		if (p instanceof TokenParser) {
			TokenParser<?, T> tp = (TokenParser<?, T>) p; 
			subpredicates.add(parse(rest[0], p.getOutputClass(), rest, tp));
		} else {
			subpredicates.add(parse(rest[0], p.getOutputClass(), rest));
		}
		p.setSubpredicates(subpredicates);
	}
	@Override
	public String toString() {
		return String.format("Filter: %s", expression);
	}
}
