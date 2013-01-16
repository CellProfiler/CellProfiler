/**
 * 
 */
package org.cellprofiler.imageset;


/**
 * @author Lee Kamentsky
 *
 * A metadata extractor router that routes the ImageFile's
 * file name to child extractors.
 */
public class FileNameMetadataExtractor extends
		MetadataExtractorAdapter<ImageFile, String> {

	public FileNameMetadataExtractor(MetadataExtractor<String> subExtractor) {
		super(subExtractor);
	}

	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractorAdapter#get(java.lang.Object)
	 */
	@Override
	protected String get(ImageFile source) {
		return source.getFileName();
	}

}
