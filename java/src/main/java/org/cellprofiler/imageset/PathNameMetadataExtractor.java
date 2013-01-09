/**
 * 
 */
package org.cellprofiler.imageset;

/**
 * @author Lee Kamentsky
 *
 * This extractor router passes the path name to its children.
 * 
 */
public class PathNameMetadataExtractor extends
		MetadataExtractorAdapter<ImageFile, String> {

	protected PathNameMetadataExtractor(MetadataExtractor<String> subExtractor) {
		super(subExtractor);
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractorAdapter#get(java.lang.Object)
	 */
	@Override
	protected String get(ImageFile source) {
		return source.getPathName();
	}

}
