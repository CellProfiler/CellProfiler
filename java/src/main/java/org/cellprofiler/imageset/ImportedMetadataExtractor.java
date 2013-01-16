/**
 * 
 */
package org.cellprofiler.imageset;

import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.cellprofiler.imageset.filter.ImagePlaneDetails;

import au.com.bytecode.opencsv.CSVReader;

/**
 * @author Lee Kamentsky
 * 
 * A metadata extractor that matches metadata entries in a .csv file
 * to those in an IPD.
 *
 */
public class ImportedMetadataExtractor implements MetadataExtractor<ImagePlaneDetails> {

	/**
	 * @author Lee Kamentsky
	 *
	 * An ImportedMetadataExtractor.KeyPair represents a pair
	 * of metadata keys used for matching. The ImportedMetadataExtractor
	 * matches sets of values in the CSV to sets of values in the
	 * ImagePlaneDetails's metadata. The metadata key names in the
	 * CSV can be different than in the IPD, for instance "PlateName"
	 * in the CSV and "Plate" in the IPD. The KeyPair allows arbitrary
	 * matching of metadata keys between the two.
	 */
	static public class KeyPair {
		final public String csvKey;
		final public String ipdKey;
		public KeyPair(String csvKey, String ipdKey){
			this.csvKey = csvKey;
			this.ipdKey = ipdKey;
		}
	}
	
	/**
	 * The CSV metadata is stored in a map of matching values to metadata to be added.
	 */
	final private Map<List<String>, Map<String, String>> importedMetadata = 
		new HashMap<List<String>, Map<String, String>>();
	/**
	 * The keys to be matched against IPD metadata, in the order they appear
	 * in the importedMetadata map key list.
	 */
	final private String [] matchingKeys;
	/**
	 * true to use case-insensitive matching
	 */
	final private boolean caseInsensitive;
	
	/**
	 * Constructor: initialize the ImportedMetadataExtractor by
	 * passing it a reader of the .csv file and an array of keys
	 * in the .csv file header that are to be used to match IPD
	 * metadata against the metadata values in the file.
	 * 
	 * @param rdr a reader of the csv file
	 * @param matchingKeys the keys to use to match rows in the CSV against rows in the ipd metadata.
	 * @param caseInsensitive true if case-insensitive matching should be used when matching values
	 * @throws IOException 
	 */
	public ImportedMetadataExtractor(Reader rdr, KeyPair [] matchingKeys, boolean caseInsensitive) 
	throws IOException {
		this.matchingKeys = new String[matchingKeys.length];
		for (int i=0; i<matchingKeys.length; i++) {
			this.matchingKeys[i] = matchingKeys[i].ipdKey;
		}
		this.caseInsensitive = caseInsensitive;
		CSVReader csvReader = new CSVReader(rdr);
		String [] allKeys = csvReader.readNext();
		if (allKeys == null)
			throw new IOException("The CSV file has no header line");
		int [] matchingPositions = new int [matchingKeys.length];
		Arrays.fill(matchingPositions, -1);
		String [] otherKeys = new String [allKeys.length - matchingKeys.length];
		int [] otherPositions = new int[otherKeys.length];
		int otherIdx = 0;
		key_loop:
		for (int kidx = 0; kidx < allKeys.length; kidx++) {
			String key = allKeys[kidx];
			for (int i=0; i<this.matchingKeys.length; i++) {
				if (key.equals(matchingKeys[i].csvKey)) {
					if (matchingPositions[i] != -1) {
						throw new IOException("Duplicate key in CSV header: " + key);
					}
					matchingPositions[i] = kidx;
					continue key_loop;
				}
			}
			if (otherIdx >= otherPositions.length) {
				/* Exception logic handled outside of loop:
				 * some matching key will not be set.
				 */
				break;
			}
			otherKeys[otherIdx] = key;
			otherPositions[otherIdx++] = kidx;
		}
		for (int i=0; i<matchingPositions.length; i++) {
			if (matchingPositions[i] == -1) {
				throw new IOException(String.format("Key, \"%s\", is missing from CSV header.", matchingKeys[i]));
			}
		}
		int line = 0;
		while(true) {
			String [] fields = csvReader.readNext();
			line++;
			if (fields == null) break;
			if (fields.length < allKeys.length) {
				throw new IOException(String.format("Line # %d: only %d values defined, expected %d", line, fields.length, allKeys.length));
			}
			final ArrayList<String> key = new ArrayList<String>(matchingKeys.length);
			for (int i=0; i<matchingPositions.length; i++) {
				if (caseInsensitive) {
					key.add(fields[matchingPositions[i]].toLowerCase());
				} else {
					key.add(fields[matchingPositions[i]]);
				}
			}
			final Map<String, String> values = new HashMap<String, String>();
			for (int i=0; i<otherPositions.length; i++) {
				values.put(otherKeys[i], fields[otherPositions[i]]);
			}
			importedMetadata.put(key, values);
		}
	}
	/* (non-Javadoc)
	 * @see org.cellprofiler.imageset.MetadataExtractor#extract(java.lang.Object)
	 */
	public Map<String, String> extract(ImagePlaneDetails source) {
		final ArrayList<String> key = new ArrayList<String>(matchingKeys.length);
		for (int i=0; i<matchingKeys.length; i++) {
			if (! source.metadata.containsKey(matchingKeys[i]))
				return emptyMap;
			if (caseInsensitive) {
				key.add(source.metadata.get(matchingKeys[i]).toLowerCase());
			} else {
				key.add(source.metadata.get(matchingKeys[i]));
			}
		}
		if (importedMetadata.containsKey(key))
			return importedMetadata.get(key);
		return emptyMap;
	}
}
