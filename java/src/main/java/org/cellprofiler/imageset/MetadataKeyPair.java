package org.cellprofiler.imageset;

import java.text.Collator;
import java.util.Comparator;

/**
 * @author Lee Kamentsky
 *
 * An ImportedMetadataExtractor.KeyPair represents a pair
 * of metadata keys used for matching. The ImportedMetadataExtractor
 * matches sets of values in the CSV to sets of values in the
 * ImagePlaneDetails's metadata. The metadata key names in the
 * CSV can be different than in the IPD, for instance "PlateName"
 * in the CSV and "Plate" in the IPD. The MetadataKeyPair allows arbitrary
 * matching of metadata keys between the two.
 */
public class MetadataKeyPair {
	final public String leftKey;
	final public String rightKey;
	final public Comparator<String> comparator;
	public MetadataKeyPair(String csvKey, String ipdKey, Comparator<String> comparator){
		this.leftKey = csvKey;
		this.rightKey = ipdKey;
		this.comparator = comparator;
	}
	/**
	 * Create a comparator of strings from one of objects
	 * @param c
	 * @return
	 */
	static private Comparator<String> adapt(final Comparator<Object> c) {
		return new Comparator<String> () {

			public int compare(String o1, String o2) {
				return c.compare(o1, o2);
			}
			
		};
	}
	static public Comparator<String> getCaseSensitiveComparator() {
		Collator c = Collator.getInstance();
		c.setStrength(Collator.IDENTICAL);
		return adapt(c);
	}
	static public Comparator<String> getCaseInsensitiveComparator() {
		Collator c = Collator.getInstance();
		c.setStrength(Collator.SECONDARY);
		return adapt(c);
	}
	static public Comparator<String> getNumericComparator() {
		return  new Comparator<String> () {
			public int compare(String o1, String o2) {
				if (o1.equals(o2)) return 0;
				return Double.valueOf(o1).compareTo(Double.valueOf(o2));
			}
		};

	}
	/**
	 * Make a key pair that requires exact case comparison
	 * @param leftKey
	 * @param rightKey
	 * @return
	 */
	static public MetadataKeyPair makeCaseSensitiveKeyPair(String csvKey, String ipdKey) {
		return new MetadataKeyPair(csvKey, ipdKey, getCaseSensitiveComparator());
	}
	/**
	 * Make a key pair that allows case-insensitive comparison
	 * @param leftKey
	 * @param rightKey
	 * @return
	 */
	static public MetadataKeyPair makeCaseInsensitiveKeyPair(String csvKey, String ipdKey) {
		return new MetadataKeyPair(csvKey, ipdKey, getCaseInsensitiveComparator());
	}
	/**
	 * Make a key pair that does numeric comparison
	 * @param leftKey
	 * @param rightKey
	 * @return
	 */
	static public MetadataKeyPair makeNumericKeyPair(String csvKey, String ipdKey) {
		return new MetadataKeyPair(csvKey, ipdKey, getNumericComparator());
	}
}