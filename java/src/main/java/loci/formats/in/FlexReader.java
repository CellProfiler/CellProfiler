/*
 * #%L
 * OME Bio-Formats package for reading and converting biological file formats.
 * %%
 * Copyright (C) 2005 - 2013 Open Microscopy Environment:
 *   - Board of Regents of the University of Wisconsin-Madison
 *   - Glencoe Software, Inc.
 *   - University of Dundee
 * %%
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 2 of the 
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public 
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/gpl-2.0.html>.
 * #L%
 */

package loci.formats.in;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Vector;

import loci.common.Constants;
import loci.common.DataTools;
import loci.common.DateTools;
import loci.common.IRandomAccess;
import loci.common.Location;
import loci.common.NIOFileHandle;
import loci.common.RandomAccessInputStream;
import loci.common.xml.BaseHandler;
import loci.common.xml.XMLTools;
import loci.formats.CoreMetadata;
import loci.formats.FormatException;
import loci.formats.FormatReader;
import loci.formats.FormatTools;
import loci.formats.MetadataTools;
import loci.formats.meta.MetadataStore;
import loci.formats.tiff.IFD;
import loci.formats.tiff.IFDList;
import loci.formats.tiff.TiffCompression;
import loci.formats.tiff.TiffConstants;
import loci.formats.tiff.TiffParser;

import ome.xml.model.primitives.NonNegativeInteger;
import ome.xml.model.primitives.PositiveFloat;
import ome.xml.model.primitives.PositiveInteger;
import ome.xml.model.primitives.Timestamp;

import org.xml.sax.Attributes;
import org.xml.sax.helpers.DefaultHandler;

/**
 * FlexReader is a file format reader for Evotec Flex files.
 * To use it, the LuraWave decoder library, lwf_jsdk2.6.jar, must be available,
 * and a LuraWave license key must be specified in the lurawave.license system
 * property (e.g., <code>-Dlurawave.license=XXXX</code> on the command line).
 *
 * <dl><dt><b>Source code:</b></dt>
 * <dd><a href="http://trac.openmicroscopy.org.uk/ome/browser/bioformats.git/components/bio-formats/src/loci/formats/in/FlexReader.java">Trac</a>,
 * <a href="http://git.openmicroscopy.org/?p=bioformats.git;a=blob;f=components/bio-formats/src/loci/formats/in/FlexReader.java;hb=HEAD">Gitweb</a></dd></dl>
 */
public class FlexReader extends FormatReader {

  // -- Constants --

  /** Custom IFD entry for Flex XML. */
  public static final int FLEX = 65200;

  public static final String FLEX_SUFFIX = "flex";
  public static final String MEA_SUFFIX = "mea";
  public static final String RES_SUFFIX = "res";
  public static final String[] MEASUREMENT_SUFFIXES =
    new String[] {MEA_SUFFIX, RES_SUFFIX};
  public static final String[] SUFFIXES =
    new String[] {FLEX_SUFFIX, MEA_SUFFIX, RES_SUFFIX};

  public static final String SCREENING = "Screening";
  public static final String ARCHIVE = "Archive";

  // -- Static fields --

  /**
   * Mapping from server names stored in the .mea file to actual server names.
   */
  private static HashMap<String, String[]> serverMap =
    new HashMap<String, String[]>();

  // -- Fields --

  /** Camera binning values. */
  private int binX, binY;

  private int plateCount;
  private int wellCount;
  private int fieldCount;

  private int wellRows, wellColumns;

  private String[] channelNames;
  private Vector<Double> xPositions, yPositions;
  private Vector<Double> xSizes, ySizes;
  private Vector<String> cameraIDs, objectiveIDs, lightSourceIDs;
  private HashMap<String, Vector<String>> lightSourceCombinationIDs;
  private Vector<String> cameraRefs, binnings, objectiveRefs;
  private Vector<String> lightSourceCombinationRefs;
  private Vector<String> filterSets;
  private HashMap<String, FilterGroup> filterSetMap;
  private HashMap<Integer, Timestamp> acquisitionDates;

  private Vector<String> measurementFiles;

  private String plateName, plateBarcode;
  private int nRows = 0, nCols = 0;

  private String plateAcqStartTime;

  private ArrayList<Double> planePositionX = new ArrayList<Double>();
  private ArrayList<Double> planePositionY = new ArrayList<Double>();
  private ArrayList<Double> planePositionZ = new ArrayList<Double>();
  private ArrayList<Double> planeExposureTime = new ArrayList<Double>();
  private ArrayList<Double> planeDeltaT = new ArrayList<Double>();

  private ArrayList<FlexFile> flexFiles;

  private int nFiles = 0;
  private int effectiveFieldCount = 0;

  /** Specifies the row and column index into 'flexFiles' for a given well. */
  private int[][] wellNumber;

  // -- Constructor --

  /** Constructs a new Flex reader. */
  public FlexReader() {
    super("Evotec Flex", SUFFIXES);
    domains = new String[] {FormatTools.HCS_DOMAIN};
    hasCompanionFiles = true;
    datasetDescription = "One directory containing one or more .flex files, " +
      "and an optional directory containing an .mea and .res file. The .mea " +
      "and .res files may also be in the same directory as the .flex file(s).";
  }

  // -- IFormatReader API methods --

  /* @see loci.formats.IFormatReader#fileGroupOption(String) */
  public int fileGroupOption(String id) throws FormatException, IOException {
    return FormatTools.MUST_GROUP;
  }

  /* @see loci.formats.IFormatReader#isSingleFile(String) */
  public boolean isSingleFile(String id) throws FormatException, IOException {
    if (!checkSuffix(id, FLEX_SUFFIX)) return false;
    return serverMap.size() == 0 || !isGroupFiles();
  }

  /* @see loci.formats.IFormatReader#getSeriesUsedFiles(boolean) */
  public String[] getSeriesUsedFiles(boolean noPixels) {
    FormatTools.assertId(currentId, true, 1);
    Vector<String> files = new Vector<String>();
    files.addAll(measurementFiles);
    if (!noPixels) {
      if (fieldCount > 0 && wellCount > 0 && plateCount > 0) {
        FlexFile file = lookupFile(getSeries());
        if (file != null && file.file != null) {
          files.add(file.file);
        }
      }
    }
    return files.toArray(new String[files.size()]);
  }

  /* @see loci.formats.IFormatReader#getOptimalTileWidth() */
  public int getOptimalTileWidth() {
    FormatTools.assertId(currentId, true, 1);

    FlexFile file = lookupFile(0);
    IFD ifd = file.ifds.get(0);
    try {
      return (int) ifd.getTileWidth();
    }
    catch (FormatException e) {
      LOGGER.debug("Could not retrieve tile width", e);
    }
    return super.getOptimalTileWidth();
  }

  /* @see loci.formats.IFormatReader#getOptimalTileHeight() */
  public int getOptimalTileHeight() {
    FormatTools.assertId(currentId, true, 1);

    FlexFile file = lookupFile(0);
    IFD ifd = file.ifds.get(0);
    try {
      return (int) ifd.getTileLength();
    }
    catch (FormatException e) {
      LOGGER.debug("Could not retrieve tile height", e);
    }
    return super.getOptimalTileHeight();
  }

  /**
   * @see loci.formats.IFormatReader#openBytes(int, byte[], int, int, int, int)
   */
  public byte[] openBytes(int no, byte[] buf, int x, int y, int w, int h)
    throws FormatException, IOException
  {
    FormatTools.checkPlaneParameters(this, no, buf.length, x, y, w, h);

    FlexFile firstFile = lookupFile(0);
    FlexFile file = lookupFile(getSeries());

    int[] lengths =
      new int[] {fieldCount / effectiveFieldCount, wellCount, plateCount};
    int[] pos = FormatTools.rasterToPosition(lengths, getSeries());

    int imageNumber = file.offsets == null ? getImageCount() * pos[0] + no : 0;

    RandomAccessInputStream s = 
      new RandomAccessInputStream(getFileHandle(file.file));

    IFD ifd;
    double factor;
    if (file.offsets == null) {
    	ifd = file.ifds.get(imageNumber);
    	factor = 1d;
    } else {
    	// Only the first IFD was read. Hack the IFD to adjust the offset.
    	final IFD firstIFD = firstFile.ifds.get(0);
    	ifd = new IFD(firstIFD);
    	int tag = IFD.STRIP_OFFSETS;
    	if (firstIFD.isTiled() && firstIFD.getIFDLongArray(IFD.TILE_OFFSETS) != null)
    		tag =  IFD.TILE_OFFSETS;
    	long [] offsets = ifd.getIFDLongArray(tag);

    	final int planeSize = getSizeX() * getSizeY() * getRGBChannelCount() *
    	ifd.getBitsPerSample()[0] / 8;
    	final int index = getImageCount() * pos[0] + no;
    	long offset = (index == file.offsets.length - 1 ?
    			s.length() : file.offsets[index + 1]) - offsets[0] - planeSize;

    	for (int i = 0; i < offsets.length; i++) {
    		offsets[i] += offset;  
    	}
    	ifd.putIFDValue(tag, offsets);
    }
    int nBytes = ifd.getBitsPerSample()[0] / 8;
    int bpp = FormatTools.getBytesPerPixel(getPixelType());

    // read pixels from the file
    TiffParser tp = new TiffParser(s);
    tp.getSamples(ifd, buf, x, y, w, h);
    factor = file.factors[imageNumber];
    tp.getStream().close();

    // expand pixel values with multiplication by factor[no]
    int num = buf.length / bpp;

    if (factor != 1d || nBytes != bpp) {
    	for (int i=num-1; i>=0; i--) {
    		int q = nBytes == 1 ? buf[i] & 0xff :
    			DataTools.bytesToInt(buf, i * nBytes, nBytes, isLittleEndian());
    		q = (int) (q * factor);
    		DataTools.unpackBytes(q, buf, i * bpp, bpp, isLittleEndian());
    	}
    }

    s.close();

    return buf;
  }

  /* @see loci.formats.IFormatReader#close(boolean) */
  public void close(boolean fileOnly) throws IOException {
    super.close(fileOnly);
    if (!fileOnly) {
      binX = binY = 0;
      plateCount = wellCount = fieldCount = 0;
      channelNames = null;
      measurementFiles = null;
      xSizes = ySizes = null;
      cameraIDs = objectiveIDs = lightSourceIDs = null;
      lightSourceCombinationIDs = null;
      lightSourceCombinationRefs = null;
      cameraRefs = objectiveRefs = binnings = null;
      wellRows = wellColumns = 0;
      xPositions = yPositions = null;
      filterSets = null;
      filterSetMap = null;
      plateName = plateBarcode = null;
      nRows = nCols = 0;
      flexFiles = null;
      wellNumber = null;
      planePositionX.clear();
      planePositionY.clear();
      planePositionZ.clear();
      planeExposureTime.clear();
      planeDeltaT.clear();
      acquisitionDates = null;
    }
  }

  // -- Internal FormatReader API methods --

  /* @see loci.formats.FormatReader#initFile(String) */
  protected void initFile(String id) throws FormatException, IOException {
    super.initFile(id);

    measurementFiles = new Vector<String>();
    acquisitionDates = new HashMap<Integer, Timestamp>();

    if (checkSuffix(id, FLEX_SUFFIX)) {
      initFlexFile(id);
    }
    else if (checkSuffix(id, RES_SUFFIX)) {
      initResFile(id);
    }
    else initMeaFile(id);

    if (plateCount == flexFiles.size()) {
      plateCount /= wellCount;
      if ((plateCount % fieldCount) == 0) {
        plateCount /= fieldCount;
      }
    }
  }

  // -- Helper methods --

  /** Initialize the dataset from a .res file. */
  private void initResFile(String id) throws FormatException, IOException {
    LOGGER.debug("initResFile({})", id);

    parseResFile(id);

    Location thisFile = new Location(id).getAbsoluteFile();
    Location parent = thisFile.getParentFile();
    LOGGER.debug("  Looking for an .mea file in {}", parent.getAbsolutePath());
    String[] list = parent.list();
    for (String file : list) {
      if (checkSuffix(file, MEA_SUFFIX)) {
        String mea = new Location(parent, file).getAbsolutePath();
        LOGGER.debug("  Found .mea file {}", mea);
        initMeaFile(mea);
        if (!measurementFiles.contains(thisFile.getAbsolutePath())) {
          measurementFiles.add(thisFile.getAbsolutePath());
        }
        return;
      }
    }
    throw new FormatException("Could not find an .mea file.");
  }

  /** Initialize the dataset from a .mea file. */
  private void initMeaFile(String id) throws FormatException, IOException {
    LOGGER.debug("initMeaFile({})", id);
    Location file = new Location(id).getAbsoluteFile();
    if (!measurementFiles.contains(file.getAbsolutePath())) {
      measurementFiles.add(file.getAbsolutePath());
    }

    // parse the .mea file to get a list of .flex files
    MeaHandler handler = new MeaHandler();
    LOGGER.info("Reading contents of .mea file");
    LOGGER.info("Parsing XML from .mea file");
    RandomAccessInputStream s = new RandomAccessInputStream(id);
    XMLTools.parseXML(s, handler);
    s.close();

    Vector<String> flex = handler.getFlexFiles();
    if (flex.size() == 0) {
      LOGGER.debug("Could not build .flex list from .mea.");
      LOGGER.info("Building list of valid .flex files");
      String[] files = findFiles(file);
      if (files != null) {
        for (String f : files) {
          if (checkSuffix(f, FLEX_SUFFIX)) flex.add(f);
        }
      }
      if (flex.size() == 0) {
        throw new FormatException(".flex files were not found. " +
          "Did you forget to specify the server names?");
      }
    }

    LOGGER.info("Looking for corresponding .res file");
    String[] files = findFiles(file, new String[] {RES_SUFFIX});
    if (files != null) {
      for (String f : files) {
        if (!measurementFiles.contains(f)) {
          measurementFiles.add(f);
        }
        parseResFile(f);
      }
    }

    MetadataStore store = makeFilterMetadata();

    groupFiles(flex.toArray(new String[flex.size()]), store);
    populateMetadataStore(store);
  }

  private void initFlexFile(String id) throws FormatException, IOException {
    LOGGER.debug("initFlexFile({})", id);
    boolean doGrouping = true;

    Location currentFile = new Location(id).getAbsoluteFile();

    LOGGER.info("Storing well indices");
    try {
      String name = currentFile.getName();
      int[] well = getWell(name);
      if (well[0] > nRows) nRows = well[0];
      if (well[1] > nCols) nCols = well[1];
    }
    catch (NumberFormatException e) {
      LOGGER.debug("Could not parse well indices", e);
      doGrouping = false;
    }

    LOGGER.info("Looking for other .flex files");
    if (!isGroupFiles()) doGrouping = false;

    if (isGroupFiles()) {
      LOGGER.debug("Attempting to find files in the same dataset.");
      try {
        findFiles(currentFile);
      }
      catch (NullPointerException e) {
        LOGGER.debug("", e);
      }
      catch (IOException e) {
        LOGGER.debug("", e);
      }
      if (measurementFiles.size() == 0) {
        LOGGER.warn("Measurement files not found.");
      }
      else {
        for (String f : measurementFiles) {
          if (checkSuffix(f, RES_SUFFIX)) {
            parseResFile(f);
          }
        }
      }
    }

    MetadataStore store = makeFilterMetadata();

    LOGGER.info("Making sure that all .flex files are valid");
    Vector<String> flex = new Vector<String>();
    if (doGrouping) {
      // group together .flex files that are in the same directory

      Location dir = currentFile.getParentFile();
      String[] files = dir.list(true);

      for (String file : files) {
        // file names should be nnnnnnnnn.flex, where 'n' is 0-9
        LOGGER.debug("Checking if {} belongs in the same dataset.", file);
        if (file.endsWith(".flex") && file.length() == 14) {
          flex.add(new Location(dir, file).getAbsolutePath());
          LOGGER.debug("Added {} to dataset.", flex.get(flex.size() - 1));
        }
      }
    }

    String[] files = doGrouping ? flex.toArray(new String[flex.size()]) :
      new String[] {currentFile.getAbsolutePath()};
    if (files.length == 0) {
      if (Location.getMappedFile(currentFile.getName()) != null) {
        files = new String[] {currentFile.getName()};
      }
      else {
        files = new String[] {currentFile.getAbsolutePath()};
      }
    }
    LOGGER.debug("Determined that {} .flex files belong together.",
      files.length);

    groupFiles(files, store);
    populateMetadataStore(store);
  }

  private void populateMetadataStore(MetadataStore store) throws FormatException
  {
    LOGGER.info("Populating MetadataStore");
    MetadataTools.populatePixels(store, this, true);

    Location currentFile = new Location(getCurrentFile()).getAbsoluteFile();
    int[] lengths = new int[] {fieldCount, wellCount, plateCount};

    store.setPlateID(MetadataTools.createLSID("Plate", 0), 0);
    String plateAcqID = MetadataTools.createLSID("PlateAcquisition", 0, 0);
    store.setPlateAcquisitionID(plateAcqID, 0, 0);

    PositiveInteger maxFieldCount = FormatTools.getMaxFieldCount(fieldCount);
    if (maxFieldCount != null) {
      store.setPlateAcquisitionMaximumFieldCount(maxFieldCount, 0, 0);
    }

    plateAcqStartTime =
      DateTools.formatDate(plateAcqStartTime, "dd.MM.yyyy  HH:mm:ss");

    if (plateAcqStartTime != null) {
      store.setPlateAcquisitionStartTime(
        new Timestamp(plateAcqStartTime), 0, 0);
    }

    for (int row=0; row<wellRows; row++) {
      for (int col=0; col<wellColumns; col++) {
        int well = row * wellColumns + col;
        store.setWellID(MetadataTools.createLSID("Well", 0, well), 0, well);
        store.setWellRow(new NonNegativeInteger(row), 0, well);
        store.setWellColumn(new NonNegativeInteger(col), 0, well);
      }
    }

    for (int i=0; i<getSeriesCount(); i++) {
      int[] pos = FormatTools.rasterToPosition(lengths, i);
      String imageID = MetadataTools.createLSID("Image", i);
      store.setImageID(imageID, i);

      int well = wellNumber[pos[1]][0] * wellColumns + wellNumber[pos[1]][1];

      char wellRow = (char) ('A' + wellNumber[pos[1]][0]);
      store.setImageName("Well " + wellRow + "-" + (wellNumber[pos[1]][1] + 1) +
        "; Field #" + (pos[0] + 1), i);

      if (acquisitionDates.get(i) != null) {
        store.setImageAcquisitionDate(acquisitionDates.get(i), i);
      }

      if (wellRows == 0 && wellColumns == 0) {
        well = pos[1];
        NonNegativeInteger row = new NonNegativeInteger(wellNumber[pos[1]][0]);
        NonNegativeInteger col = new NonNegativeInteger(wellNumber[pos[1]][1]);
        String wellID = MetadataTools.createLSID("Well", pos[2], well);
        store.setWellID(wellID, pos[2], well);
        store.setWellRow(row, pos[2], pos[1]);
        store.setWellColumn(col, pos[2], pos[1]);
      }
      String wellSample =
        MetadataTools.createLSID("WellSample", pos[2], well, pos[0]);
      store.setWellSampleID(wellSample, pos[2], well, pos[0]);
      store.setWellSampleIndex(new NonNegativeInteger(i), pos[2], well, pos[0]);
      store.setWellSampleImageRef(imageID, pos[2], well, pos[0]);

      store.setPlateAcquisitionWellSampleRef(wellSample, 0, 0, i);
    }

    if (getMetadataOptions().getMetadataLevel() != MetadataLevel.MINIMUM) {
      String instrumentID = MetadataTools.createLSID("Instrument", 0);
      store.setInstrumentID(instrumentID, 0);

      if (plateName == null) plateName = currentFile.getParentFile().getName();
      if (plateBarcode != null) plateName = plateBarcode + " " + plateName;
      store.setPlateName(plateName, 0);
      store.setPlateRowNamingConvention(getNamingConvention("Letter"), 0);
      store.setPlateColumnNamingConvention(getNamingConvention("Number"), 0);

      for (int i=0; i<getSeriesCount(); i++) {
        int[] pos = FormatTools.rasterToPosition(lengths, i);

        store.setImageInstrumentRef(instrumentID, i);

        int seriesIndex = i * getImageCount();
        if (seriesIndex < objectiveRefs.size()) {
          store.setObjectiveSettingsID(objectiveRefs.get(seriesIndex), i);
        }

        for (int c=0; c<getEffectiveSizeC(); c++) {
          int channelIndex = seriesIndex + c;
          if (channelNames != null && channelIndex < channelNames.length) {
            store.setChannelName(channelNames[channelIndex], i, c);
          }
        }

        if (seriesIndex < lightSourceCombinationRefs.size()) {
          String lightSourceCombo = lightSourceCombinationRefs.get(seriesIndex);
          Vector<String> lightSources =
            lightSourceCombinationIDs.get(lightSourceCombo);

          for (int c=0; c<getEffectiveSizeC(); c++) {
            int index = seriesIndex + c;
            if (index < cameraRefs.size()) {
              store.setDetectorSettingsID(cameraRefs.get(index), i, c);
              if (index < binnings.size()) {
                store.setDetectorSettingsBinning(
                  getBinning(binnings.get(index)), i, c);
              }
            }
            if (lightSources != null && c < lightSources.size()) {
              store.setChannelLightSourceSettingsID(lightSources.get(c), i, c);
            }
            else if (c > 0 && lightSources != null && lightSources.size() == 1)
            {
              lightSourceCombo =
                lightSourceCombinationRefs.get(seriesIndex + c);
              lightSources = lightSourceCombinationIDs.get(lightSourceCombo);
              store.setChannelLightSourceSettingsID(lightSources.get(0), i, c);
            }
            if (index < filterSets.size()) {
              FilterGroup group = filterSetMap.get(filterSets.get(index));
              if (group.emission != null) {
                store.setLightPathEmissionFilterRef(group.emission, i, c, 0);
              }
              if (group.excitation != null) {
                store.setLightPathExcitationFilterRef(
                  group.excitation, i, c, 0);
              }
              if (group.dichroic != null) {
                store.setLightPathDichroicRef(group.dichroic, i, c);
              }
            }
          }
        }

        if (seriesIndex < xSizes.size()) {
          PositiveFloat size =
            FormatTools.getPhysicalSizeX(xSizes.get(seriesIndex));
          if (size != null) {
            store.setPixelsPhysicalSizeX(size, i);
          }
        }
        if (seriesIndex < ySizes.size()) {
          PositiveFloat size =
            FormatTools.getPhysicalSizeY(ySizes.get(seriesIndex));
          if (size != null) {
            store.setPixelsPhysicalSizeY(size, i);
          }
        }

        int well = wellNumber[pos[1]][0] * wellColumns + wellNumber[pos[1]][1];
        if (wellRows == 0 && wellColumns == 0) {
          well = pos[1];
        }

        if (pos[0] < xPositions.size()) {
          store.setWellSamplePositionX(
            xPositions.get(pos[0]), pos[2], well, pos[0]);
        }
        if (pos[0] < yPositions.size()) {
          store.setWellSamplePositionY(
            yPositions.get(pos[0]), pos[2], well, pos[0]);
        }

        for (int image=0; image<getImageCount(); image++) {
          int plane = i * getImageCount() + image;
          int c = getZCTCoords(image)[1];
          if (plane < planePositionX.size()) {
            store.setPlanePositionX(planePositionX.get(plane), i, image);
          }
          if (plane < planePositionY.size()) {
            store.setPlanePositionY(planePositionY.get(plane), i, image);
          }
          if (plane < planePositionZ.size()) {
            store.setPlanePositionZ(planePositionZ.get(plane), i, image);
          }
          if (plane - image + c < planeExposureTime.size()) {
            store.setPlaneExposureTime(
              planeExposureTime.get(plane - image + c), i, image);
          }
          if (plane < planeDeltaT.size()) {
            store.setPlaneDeltaT(planeDeltaT.get(plane), i, image);
          }
        }
      }
    }
  }

  private void parseResFile(String id) throws IOException {
    ResHandler handler = new ResHandler();
    String resXML = DataTools.readFile(id);
    XMLTools.parseXML(resXML, handler);
  }

  /**
   * Returns a two-element array containing the well row and well column
   * corresponding to the given file.
   */
  private int[] getWell(String file) {
    String name = file.substring(file.lastIndexOf(File.separator) + 1);
    if (name.length() == 14) {
      // expect nnnnnnnnn.flex
      try {
        int row = Integer.parseInt(name.substring(0, 3)) - 1;
        int col = Integer.parseInt(name.substring(3, 6)) - 1;
        return new int[] {row, col};
      }
      catch (NumberFormatException e) { }
    }
    return new int[] {0, 0};
  }

  private int getField(String file) {
    String name = file.substring(file.lastIndexOf(File.separator) + 1);
    if (name.length() == 14) {
      // expect nnnnnnnnn.flex
      try {
        return Integer.parseInt(name.substring(6, 9)) - 1;
      }
      catch (NumberFormatException e) { }
    }
    return 0;
  }

  /**
   * Returns the number of planes in the first well that has data. May not be
   * <code>[0][0]</code> as the acquisition may have been column or row offset.
   */
  private int firstWellPlanes() {
    for (FlexFile file : flexFiles) {
      if (file.offsets != null) {
        return file.offsets.length;
      }
      if (file.ifds != null) {
        return file.ifds.size();
      }
    }
    return 0;
  }

  /**
   * Parses XML metadata from the Flex file corresponding to the given well.
   * If the 'firstFile' flag is set, then the core metadata is also
   * populated.
   */
  private void parseFlexFile(int currentWell, int wellRow, int wellCol,
    int field, boolean firstFile, MetadataStore store)
    throws FormatException, IOException
  {
    LOGGER.info("Parsing .flex file (well {}{})",
      (char) (wellRow + 'A'), wellCol + 1);
    FlexFile file = lookupFile(wellRow, wellCol, field);
    if (file == null) return;

    int originalFieldCount = fieldCount;

    if (xPositions == null) xPositions = new Vector<Double>();
    if (yPositions == null) yPositions = new Vector<Double>();
    if (xSizes == null) xSizes = new Vector<Double>();
    if (ySizes == null) ySizes = new Vector<Double>();
    if (cameraIDs == null) cameraIDs = new Vector<String>();
    if (lightSourceIDs == null) lightSourceIDs = new Vector<String>();
    if (objectiveIDs == null) objectiveIDs = new Vector<String>();
    if (lightSourceCombinationIDs == null) {
      lightSourceCombinationIDs = new HashMap<String, Vector<String>>();
    }
    if (lightSourceCombinationRefs == null) {
      lightSourceCombinationRefs = new Vector<String>();
    }
    if (cameraRefs == null) cameraRefs = new Vector<String>();
    if (objectiveRefs == null) objectiveRefs = new Vector<String>();
    if (binnings == null) binnings = new Vector<String>();
    if (filterSets == null) filterSets = new Vector<String>();
    if (filterSetMap == null) filterSetMap = new HashMap<String, FilterGroup>();

    // parse factors from XML
    LOGGER.debug("Parsing XML from {}", file.file);

    int nOffsets = file.offsets != null ? file.offsets.length : 0;

    IFD ifd = null;
    if (nOffsets == 0) {
      ifd = file.ifds.get(0);
    }
    else {
      TiffParser parser = new TiffParser(file.file);
      ifd = parser.getFirstIFD();
      parser.getStream().close();
    }
    String xml = XMLTools.sanitizeXML(ifd.getIFDStringValue(FLEX));

    Vector<String> n = new Vector<String>();
    Vector<String> f = new Vector<String>();
    DefaultHandler handler =
      new FlexHandler(n, f, store, firstFile, currentWell);
    LOGGER.info("Parsing XML in .flex file");
    XMLTools.parseXML(xml.getBytes(Constants.ENCODING), handler);

    channelNames = n.toArray(new String[n.size()]);

    if (firstFile) populateCoreMetadata(wellRow, wellCol, field, n);

    int totalPlanes = getSeriesCount() * getImageCount();

    LOGGER.info("Populating pixel scaling factors");

    // verify factor count
    int nsize = n.size();
    int fsize = f.size();
    if (nsize != fsize || nsize != totalPlanes) {
      LOGGER.warn("mismatch between image count, names and factors " +
        "(count={}, names={}, factors={})",
        new Object[] {totalPlanes, nsize, fsize});
    }
    for (String ns : n) {
      addGlobalMetaList("Name", ns);
    }
    for (String fs : f) {
      addGlobalMetaList("Factor", fs);
    }

    // parse factor values
    file.factors = new double[totalPlanes];
    int max = 0;
    for (int i=0; i<fsize; i++) {
      String factor = f.get(i);
      double q = 1;
      try {
        q = Double.parseDouble(factor);
      }
      catch (NumberFormatException exc) {
        LOGGER.warn("invalid factor #{}: {}", i, factor);
      }
      if (i < file.factors.length) {
        file.factors[i] = q;
        if (q > file.factors[max]) max = i;
      }
    }
    if (fsize < file.factors.length) {
      Arrays.fill(file.factors, fsize, file.factors.length, 1);
    }

    // determine pixel type
    if (file.factors[max] > 256) {
      core.get(0).pixelType = FormatTools.UINT32;
    }
    else if (file.factors[max] > 1) {
      core.get(0).pixelType = FormatTools.UINT16;
    }
    for (int i=1; i<core.size(); i++) {
      core.get(i).pixelType = getPixelType();
    }

    if (!firstFile) {
      fieldCount = originalFieldCount;
    }
  }

  /** Populate core metadata using the given list of image names. */
  private void populateCoreMetadata(int wellRow, int wellCol, int field,
    Vector<String> imageNames)
    throws FormatException
  {
    LOGGER.info("Populating core metadata for well row " + wellRow +
      ", column " + wellCol);
    CoreMetadata ms0 = core.get(0);
    if (getSizeC() == 0 && getSizeT() == 0) {
      if (fieldCount == 0 || (imageNames.size() % fieldCount) != 0) {
        fieldCount = 1;
      }

      Vector<String> uniqueChannels = new Vector<String>();
      for (int i=0; i<imageNames.size(); i++) {
        String name = imageNames.get(i);
        String[] tokens = name.split("_");
        if (tokens.length > 1) {
          // fields are indexed from 1
          int fieldIndex = Integer.parseInt(tokens[0]);
          if (fieldIndex > fieldCount) fieldCount = fieldIndex;
        }
        else tokens = name.split(":");
        String channel = tokens[tokens.length - 1];
        if (!uniqueChannels.contains(channel)) uniqueChannels.add(channel);
      }
      if (fieldCount == 0) fieldCount = 1;
      ms0.sizeC = (int) Math.max(uniqueChannels.size(), 1);
      if (getSizeZ() == 0) ms0.sizeZ = 1;
      ms0.sizeT =
        imageNames.size() / (fieldCount * getSizeC() * getSizeZ());
    }

    if (getSizeC() == 0) {
      ms0.sizeC = (int) Math.max(channelNames.length, 1);
    }

    if (getSizeZ() == 0) ms0.sizeZ = 1;
    if (getSizeT() == 0) ms0.sizeT = 1;
    if (plateCount == 0) plateCount = 1;
    if (wellCount == 0) wellCount = 1;
    if (fieldCount == 0) fieldCount = 1;

    // adjust dimensions if the number of IFDs doesn't match the number
    // of reported images

    FlexFile file = lookupFile(wellRow, wellCol, field);
    IFD ifd = file.ifds.get(0);
    int nPlanes = file.ifds.size();
    if (file.offsets != null) {
      nPlanes = file.offsets.length;
    }

    ms0.imageCount = getSizeZ() * getSizeC() * getSizeT();
    if (getImageCount() * fieldCount != nPlanes) {
      ms0.imageCount = nPlanes / fieldCount;
      ms0.sizeZ = 1;
      ms0.sizeT = nPlanes / fieldCount;
      if (getSizeT() % getSizeC() == 0) {
        ms0.sizeT /= getSizeC();
      }
      else {
        ms0.sizeC = 1;
      }
    }
    ms0.sizeX = (int) ifd.getImageWidth();
    ms0.sizeY = (int) ifd.getImageLength();
    ms0.dimensionOrder = "XYCZT";
    ms0.rgb = false;
    ms0.interleaved = false;
    ms0.indexed = false;
    ms0.littleEndian = ifd.isLittleEndian();
    ms0.pixelType = ifd.getPixelType();

    if (fieldCount == 1) {
      fieldCount *= nFiles;
    }

    int seriesCount = plateCount * wellCount * fieldCount;
    if (seriesCount > 1) {
      core.clear();
      for (int i = 0; i < seriesCount; i++) {
        core.add(ms0);
      }
    }
  }

  /**
   * Search for files that correspond to the given file.
   * If the given file is a .mea file, then the corresponding files will be
   * .res and .flex files.
   * If the given file is a .flex file, then the corresponding files will be
   * .res and .mea files.
   */
  private String[] findFiles(Location baseFile) throws IOException {
    String[] suffixes = new String[0];
    if (checkSuffix(baseFile.getName(), FLEX_SUFFIX)) {
      suffixes = new String[] {MEA_SUFFIX, RES_SUFFIX};
      LOGGER.debug("Looking for files with the suffix '{}' or '{}'.",
        MEA_SUFFIX, RES_SUFFIX);
    }
    else if (checkSuffix(baseFile.getName(), MEA_SUFFIX)) {
      suffixes = new String[] {FLEX_SUFFIX, RES_SUFFIX};
      LOGGER.debug("Looking for files with the suffix '{}' or '{}'.",
        FLEX_SUFFIX, RES_SUFFIX);
    }

    return findFiles(baseFile, suffixes);
  }

  private String[] findFiles(Location baseFile, String[] suffixes)
    throws IOException
  {
    // we're assuming that the directory structure looks something like this:
    //
    //                        top level directory
    //                         /              \
    //           top level flex dir       top level measurement dir
    //              /     |    \                 /       |     \
    //        plate #0   ...   plate #n     plate #0    ...    plate #n
    //       /   |  \                        /   \
    //    .flex ... .flex                 .mea   .res
    //
    // or like this:
    //
    //                       top level directory
    //                       /  |  \      /  |  \
    //          Flex plate #0  ... #n    #0 ... Measurement plate #n
    //
    // or that the .mea and .res are in the same directory as the .flex files

    LOGGER.debug("findFiles({})", baseFile.getAbsolutePath());

    LOGGER.info("Looking for files that are in the same dataset as " +
      baseFile.getAbsolutePath());
    Vector<String> fileList = new Vector<String>();

    Location plateDir = baseFile.getParentFile();
    String[] files = plateDir.list(true);

    // check if the measurement files are in the same directory
    LOGGER.debug("Looking for files in {}", plateDir.getAbsolutePath());
    for (String file : files) {
      String lfile = file.toLowerCase();
      String path = new Location(plateDir, file).getAbsolutePath();
      if (checkSuffix(file, suffixes)) {
        fileList.add(path);
        LOGGER.debug("Found file {}", path);
      }
    }

    // file list is valid (i.e. can be returned) if there is at least
    // one file with each of the desired suffixes
    LOGGER.debug(
      "Checking to see if at least one file with each suffix was found...");
    boolean validList = true;
    for (String suffix : suffixes) {
      boolean foundSuffix = false;
      for (String file : fileList) {
        if (checkSuffix(file, suffix)) {
          foundSuffix = true;
          break;
        }
      }
      if (!foundSuffix) {
        validList = false;
        break;
      }
    }
    LOGGER.debug("{} required files.", validList ? "Found" : "Did not find");

    if (validList) {
      LOGGER.debug("Returning file list:");
      for (String file : fileList) {
        LOGGER.debug("  {}", file);
        if (checkSuffix(file, MEASUREMENT_SUFFIXES) &&
          !measurementFiles.contains(file))
        {
          measurementFiles.add(file);
        }
      }
      return fileList.toArray(new String[fileList.size()]);
    }

    Location flexDir = null;
    try {
      flexDir = plateDir.getParentFile();
    }
    catch (NullPointerException e) { }
    LOGGER.debug("Looking for files in {}", flexDir);
    if (flexDir == null) return null;

    // check if the measurement directory and the Flex directory
    // have the same parent

    Location measurementDir = null;
    String[] flexDirList = flexDir.list(true);
    if (flexDirList.length > 1) {
      String plateName = plateDir.getName();
      for (String file : flexDirList) {
        if (!file.equals(plateName) &&
          (plateName.startsWith(file) || file.startsWith(plateName)))
        {
          measurementDir = new Location(flexDir, file);
          LOGGER.debug("Expect measurement files to be in {}",
            measurementDir.getAbsolutePath());
          break;
        }
      }
    }

    // check if Flex directories and measurement directories have
    // a different parent

    if (measurementDir == null) {
      Location topDir = flexDir.getParentFile();
      LOGGER.debug("First attempt at finding measurement file directory " +
        "failed.  Looking for an appropriate measurement directory in {}.",
        topDir.getAbsolutePath());

      String[] topDirList = topDir.list(true);
      for (String file : topDirList) {
        if (!flexDir.getAbsolutePath().endsWith(file)) {
          measurementDir = new Location(topDir, file);
          LOGGER.debug("Expect measurement files to be in {}",
            measurementDir.getAbsolutePath());
          break;
        }
      }

      if (measurementDir == null) {
        LOGGER.debug("Failed to find measurement file directory.");
        return null;
      }
    }
    else plateDir = measurementDir;

    if (!plateDir.getAbsolutePath().equals(measurementDir.getAbsolutePath())) {
      LOGGER.debug("Measurement files are in a subdirectory of {}",
        measurementDir.getAbsolutePath());
      String[] measurementPlates = measurementDir.list(true);
      String plate = plateDir.getName();
      LOGGER.debug("Determining which subdirectory contains the measurements " +
        "for plate {}", plate);
      plateDir = null;
      if (measurementPlates != null) {
        for (String file : measurementPlates) {
          LOGGER.debug("Checking {}", file);
          if (file.indexOf(plate) != -1 || plate.indexOf(file) != -1) {
            plateDir = new Location(measurementDir, file);
            LOGGER.debug("Measurement files are in {}",
              plateDir.getAbsolutePath());
            break;
          }
        }
      }
    }

    if (plateDir == null) {
      LOGGER.debug("Could not find appropriate subdirectory.");
      return null;
    }

    files = plateDir.list(true);
    for (String file : files) {
      fileList.add(new Location(plateDir, file).getAbsolutePath());
    }

    LOGGER.debug("Returning file list:");
    for (String file : fileList) {
      LOGGER.debug("  {}", file);
      if (checkSuffix(file, MEASUREMENT_SUFFIXES) &&
        !measurementFiles.contains(file))
      {
        measurementFiles.add(file);
      }
    }
    return fileList.toArray(new String[fileList.size()]);
  }

  private void groupFiles(String[] fileList, MetadataStore store)
    throws FormatException, IOException
  {
    LOGGER.info("Grouping together files in the same dataset");
    HashMap<String, ArrayList<String>> v = new HashMap<String, ArrayList<String>>();
    Boolean firstCompressed = null;
    int firstIFDCount = 0;
    for (String file : fileList) {
      RandomAccessInputStream s = new RandomAccessInputStream(file);
      TiffParser parser = new TiffParser(s);
      IFD firstIFD = parser.getFirstIFD();
      int ifdCount = parser.getIFDOffsets().length;
      s.close();
      boolean compressed =
        firstIFD.getCompression() != TiffCompression.UNCOMPRESSED;
      if (firstCompressed == null) {
        firstCompressed = compressed;
        firstIFDCount = ifdCount;
      }
      if (compressed == firstCompressed && ifdCount == firstIFDCount) {
        int[] well = getWell(file);
        int field = getField(file);
        if (well[0] > nRows) nRows = well[0];
        if (well[1] > nCols) nCols = well[1];
        if (fileList.length == 1) {
          well[0] = 0;
          well[1] = 0;
        }

        ArrayList<String> files = v.get(well[0] + "," + well[1]);
        if (files == null) {
          files = new ArrayList<String>();
        }
        files.add(file);
        v.put(well[0] + "," + well[1], files);
      }
      else {
        v.clear();
        ArrayList<String> files = new ArrayList<String>();
        files.add(currentId);
        v.put("0,0", files);
        fileList = new String[] {currentId};
        break;
      }
    }

    nRows++;
    nCols++;

    if (fileList.length == 1) {
      nRows = 1;
      nCols = 1;
    }

    LOGGER.debug("Determined that there are {} rows and {} columns of wells.",
      nRows, nCols);

    flexFiles = new ArrayList<FlexFile>();
    wellCount = v.size();
    wellNumber = new int[wellCount][2];

    RandomAccessInputStream s = null;
    boolean firstFile = true;
    boolean compressed = false;
    int nOffsets = 1;

    int currentWell = 0;
    for (int row=0; row<nRows; row++) {
      for (int col=0; col<nCols; col++) {
        ArrayList<String> files = v.get(row + "," + col);
        if (files == null) {
          continue;
        }

        nFiles = files.size();

        String[] sortedFiles = files.toArray(new String[files.size()]);
        Arrays.sort(sortedFiles);
        files.clear();
        for (String f : sortedFiles) {
          files.add(f);
        }

        for (int field=0; field<nFiles; field++) {
          FlexFile file = new FlexFile();
          file.row = row;
          file.column = col;
          file.field = field;
          file.file = files.get(field);

          if (file.file == null) {
            continue;
          }

          wellNumber[currentWell][0] = row;
          wellNumber[currentWell][1] = col;

          s = new RandomAccessInputStream(getFileHandle(file.file));
          TiffParser tp = new TiffParser(s);

          if (compressed || firstFile) {
            LOGGER.info("Parsing IFDs for well {}{}",
              (char) (row + 'A'), col + 1);
            IFD firstIFD = tp.getFirstIFD();
            compressed =
              firstIFD.getCompression() != TiffCompression.UNCOMPRESSED;

            if (compressed) {
              tp.setDoCaching(false);
              file.ifds = tp.getIFDs();
              file.ifds.set(0, firstIFD);
              flexFiles.add(file);
              parseFlexFile(currentWell, row, col, field, firstFile, store);
            }
            else {
              // if the pixel data is uncompressed, we can assume that
              // the pixel data for image #0 is located immediately before
              // IFD #1; as a result, we only need to parse the first IFD
              file.offsets = tp.getIFDOffsets();
              nOffsets = file.offsets.length;
              file.ifds = new IFDList();
              file.ifds.add(firstIFD);
              flexFiles.add(file);
              parseFlexFile(currentWell, row, col, field, firstFile, store);
            }
          }
          else {
            // retrieve the offsets to each IFD, instead of parsing
            // all of the IFDs
            LOGGER.info("Retrieving IFD offsets for well {}{}",
              (char) (row + 'A'), col + 1);
            file.offsets = new long[nOffsets];

            // Assume that all IFDs after the first are evenly spaced.
            // TiffParser.getIFDOffsets() could be used instead, but is
            // substantially slower.
            file.offsets[0] = tp.getFirstOffset();
            if (file.offsets.length > 1) {
              s.seek(file.offsets[0]);
              s.skipBytes(s.readShort() * TiffConstants.BYTES_PER_ENTRY);
              file.offsets[1] = s.readInt();
              int size = FormatTools.getPlaneSize(this) + 174;
              for (int i=2; i<file.offsets.length; i++) {
                file.offsets[i] = file.offsets[i - 1] + size;
              }
            }
            flexFiles.add(file);
            parseFlexFile(currentWell, row, col, field, firstFile, store);
          }
          s.close();
          if (firstFile) firstFile = false;
        }
        currentWell++;
      }
    }
  }

  private IRandomAccess getFileHandle(String flexFile) throws IOException {
    if (Location.getMappedFile(flexFile) != null) {
      return Location.getMappedFile(flexFile);
    }
    return new NIOFileHandle(flexFile, "r");
  }

  private FlexFile lookupFile(int wellRow, int wellColumn, int field) {
    for (FlexFile file : flexFiles) {
      if (file.row == wellRow && file.column == wellColumn &&
        file.field == field)
      {
        return file;
      }
    }
    return null;
  }

  private FlexFile lookupFile(int fileSeries) {
    effectiveFieldCount = fieldCount;
    if (wellCount * plateCount == flexFiles.size()) {
      effectiveFieldCount = 1;
    }

    int[] lengths = new int[] {fieldCount, wellCount, plateCount};
    int[] pos = FormatTools.rasterToPosition(lengths, fileSeries);

    boolean zeroWell = wellCount == 1 && effectiveFieldCount == 1;
    int row = zeroWell ? 0 : wellNumber[pos[1]][0];
    int col = zeroWell ? 0 : wellNumber[pos[1]][1];

    return lookupFile(row, col, effectiveFieldCount == 1 ? 0 : pos[0]);
  }

  // -- Helper classes --

  class FlexFile {
    public int row;
    public int column;
    public int field;

    public String file;
    public IFDList ifds;
    public long[] offsets;
    public double[] factors;
  }

  /** SAX handler for parsing XML. */
  public class FlexHandler extends BaseHandler {
    private Vector<String> names, factors;
    private MetadataStore store;

    private int nextLaser = -1;
    private int nextCamera = 0;
    private int nextObjective = -1;
    private int nextImage = 0;
    private int nextPlate = 0;

    private String parentQName;
    private String lightSourceID;

    private String sliderName;
    private int nextFilter;
    private int nextDichroic;
    private int nextFilterSet;
    private int nextSliderRef;

    private boolean populateCore = true;
    private int well = 0;

    private HashMap<String, String> filterMap;
    private HashMap<String, String> dichroicMap;
    private MetadataLevel level;

    private String filterSet;

    private StringBuffer charData = new StringBuffer();

    public FlexHandler(Vector<String> names, Vector<String> factors,
      MetadataStore store, boolean populateCore, int well)
    {
      this.names = names;
      this.factors = factors;
      this.store = store;
      this.populateCore = populateCore;
      this.well = well;
      filterMap = new HashMap<String, String>();
      dichroicMap = new HashMap<String, String>();
      level = getMetadataOptions().getMetadataLevel();
    }

    public void characters(char[] ch, int start, int length) {
      charData.append(new String(ch, start, length));
    }

    public void endElement(String uri, String localName, String qName) {
      String value = charData.toString();
      charData = new StringBuffer();

      if (qName.equals("XSize") && "Plate".equals(parentQName)) {
        wellRows = Integer.parseInt(value);
      }
      else if (qName.equals("YSize") && "Plate".equals(parentQName)) {
        wellColumns = Integer.parseInt(value);
      }
      else if ("Image".equals(parentQName)) {
        if (fieldCount == 0) fieldCount = 1;
        int nImages = firstWellPlanes() / fieldCount;
        if (nImages == 0) nImages = 1; // probably a manually altered dataset
        int currentSeries = (nextImage - 1) / nImages;
        currentSeries += well * fieldCount;
        int currentImage = (nextImage - 1) % nImages;

        int seriesCount = 1;
        if (plateCount > 0) seriesCount *= plateCount;
        if (wellCount > 0) seriesCount *= wellCount;
        if (fieldCount > 0) seriesCount *= fieldCount;
        if (currentSeries >= seriesCount) return;

        if (qName.equals("DateTime")) {
          if (value != null) {
            acquisitionDates.put(currentSeries, new Timestamp(value));
          }
        }
      }

      if (level == MetadataLevel.MINIMUM) return;

      if (qName.equals("Image")) {
        binnings.add(binX + "x" + binY);
      }
      else if (qName.equals("PlateName")) {
        if (plateName == null) plateName = value;
      }
      else if (qName.equals("Barcode")) {
        if (plateBarcode == null) plateBarcode = value;
        store.setPlateExternalIdentifier(value, nextPlate - 1);
      }
      else if (qName.equals("Wavelength")) {
        String lsid = MetadataTools.createLSID("LightSource", 0, nextLaser);
        store.setLaserID(lsid, 0, nextLaser);
        Integer wavelength = new Integer(value);
        PositiveInteger wave = FormatTools.getWavelength(wavelength);
        if (wave != null) {
          store.setLaserWavelength(wave, 0, nextLaser);
        }
        try {
          store.setLaserType(getLaserType("Other"), 0, nextLaser);
          store.setLaserLaserMedium(getLaserMedium("Other"), 0, nextLaser);
        }
        catch (FormatException e) {
          LOGGER.warn("", e);
        }
      }
      else if (qName.equals("Magnification")) {
        store.setObjectiveCalibratedMagnification(new Double(value), 0,
          nextObjective);
      }
      else if (qName.equals("NumAperture")) {
        store.setObjectiveLensNA(new Double(value), 0, nextObjective);
      }
      else if (qName.equals("Immersion")) {
        String immersion = "Other";
        if (value.equals("1.33")) immersion = "Water";
        else if (value.equals("1.00")) immersion = "Air";
        else LOGGER.warn("Unknown immersion medium: {}", value);
        try {
          store.setObjectiveImmersion(
            getImmersion(immersion), 0, nextObjective);
        }
        catch (FormatException e) {
          LOGGER.warn("", e);
        }
      }
      else if (qName.equals("OffsetX") || qName.equals("OffsetY")) {
        Double offset = new Double(Double.parseDouble(value) * 1000000);
        if (qName.equals("OffsetX")) xPositions.add(offset);
        else yPositions.add(offset);
      }
      else if ("Image".equals(parentQName)) {
        if (fieldCount == 0) fieldCount = 1;
        int nImages = firstWellPlanes() / fieldCount;
        if (nImages == 0) nImages = 1; // probably a manually altered dataset
        int currentSeries = (nextImage - 1) / nImages;
        currentSeries += well * fieldCount;
        int currentImage = (nextImage - 1) % nImages;

        int seriesCount = 1;
        if (plateCount > 0) seriesCount *= plateCount;
        if (wellCount > 0) seriesCount *= wellCount;
        if (fieldCount > 0) seriesCount *= fieldCount;
        if (currentSeries >= seriesCount) return;

        if (qName.equals("CameraBinningX")) {
          binX = Integer.parseInt(value);
        }
        else if (qName.equals("CameraBinningY")) {
          binY = Integer.parseInt(value);
        }
        else if (qName.equals("ObjectiveRef")) {
          String objectiveID = MetadataTools.createLSID(
            "Objective", 0, objectiveIDs.indexOf(value));
          objectiveRefs.add(objectiveID);
        }
        else if (qName.equals("CameraRef")) {
          int cameraIndex = cameraIDs.indexOf(value);
          if (cameraIndex >= 0) {
            String detectorID =
              MetadataTools.createLSID("Detector", 0, cameraIndex);
            cameraRefs.add(detectorID);
          }
        }
        else if (qName.equals("ImageResolutionX")) {
          double v = Double.parseDouble(value) * 1000000;
          xSizes.add(new Double(v));
        }
        else if (qName.equals("ImageResolutionY")) {
          double v = Double.parseDouble(value) * 1000000;
          ySizes.add(new Double(v));
        }
        else if (qName.equals("PositionX")) {
          Double v = new Double(Double.parseDouble(value) * 1000000);
          planePositionX.add(v);
          addGlobalMetaList("X position for position", v);
        }
        else if (qName.equals("PositionY")) {
          Double v = new Double(Double.parseDouble(value) * 1000000);
          planePositionY.add(v);
          addGlobalMetaList("Y position for position", v);
        }
        else if (qName.equals("PositionZ")) {
          Double v = new Double(Double.parseDouble(value) * 1000000);
          planePositionZ.add(v);
          addGlobalMetaList("Z position for position", v);
        }
        else if (qName.equals("TimepointOffsetUsed")) {
          planeDeltaT.add(new Double(value));
        }
        else if (qName.equals("CameraExposureTime")) {
          planeExposureTime.add(new Double(value));
        }
        else if (qName.equals("LightSourceCombinationRef")) {
          lightSourceCombinationRefs.add(value);
        }
        else if (qName.equals("FilterCombinationRef")) {
          filterSets.add("FilterSet:" + value);
        }
      }
      else if (qName.equals("FilterCombination")) {
        nextFilterSet++;
        nextSliderRef = 0;
      }
    }

    public void startElement(String uri,
      String localName, String qName, Attributes attributes)
    {
      if (qName.equals("Array")) {
        int len = attributes.getLength();
        for (int i=0; i<len; i++) {
          String name = attributes.getQName(i);
          if (name.equals("Name")) {
            names.add(attributes.getValue(i));
          }
          else if (name.equals("Factor")) factors.add(attributes.getValue(i));
        }
      }
      else if (qName.equals("LightSource") && level != MetadataLevel.MINIMUM) {
        parentQName = qName;
        String type = attributes.getValue("LightSourceType");

        lightSourceIDs.add(attributes.getValue("ID"));
        nextLaser++;
      }
      else if (qName.equals("LightSourceCombination") &&
        level != MetadataLevel.MINIMUM)
      {
        lightSourceID = attributes.getValue("ID");
        lightSourceCombinationIDs.put(lightSourceID, new Vector<String>());
      }
      else if (qName.equals("LightSourceRef") && level != MetadataLevel.MINIMUM)
      {
        Vector<String> v = lightSourceCombinationIDs.get(lightSourceID);
        if (v != null) {
          int id = lightSourceIDs.indexOf(attributes.getValue("ID"));
          String lightSourceID = MetadataTools.createLSID("LightSource", 0, id);
          v.add(lightSourceID);
          lightSourceCombinationIDs.put(lightSourceID, v);
        }
      }
      else if (qName.equals("Camera") && level != MetadataLevel.MINIMUM) {
        parentQName = qName;
        String detectorID = MetadataTools.createLSID("Detector", 0, nextCamera);
        store.setDetectorID(detectorID, 0, nextCamera);
        try {
          store.setDetectorType(getDetectorType(
            attributes.getValue("CameraType")), 0, nextCamera);
        }
        catch (FormatException e) {
          LOGGER.warn("", e);
        }
        cameraIDs.add(attributes.getValue("ID"));
        nextCamera++;
      }
      else if (qName.equals("Objective") && level != MetadataLevel.MINIMUM) {
        parentQName = qName;
        nextObjective++;

        String objectiveID =
          MetadataTools.createLSID("Objective", 0, nextObjective);
        store.setObjectiveID(objectiveID, 0, nextObjective);
        try {
          store.setObjectiveCorrection(
            getCorrection("Other"), 0, nextObjective);
        }
        catch (FormatException e) {
          LOGGER.warn("", e);
        }
        objectiveIDs.add(attributes.getValue("ID"));
      }
      else if (qName.equals("Field")) {
        parentQName = qName;
        int fieldNo = Integer.parseInt(attributes.getValue("No"));
        if (fieldNo > fieldCount && fieldCount < firstWellPlanes()) {
          fieldCount++;
        }
      }
      else if (qName.equals("Plane")) {
        parentQName = qName;
        int planeNo = Integer.parseInt(attributes.getValue("No"));
        if (planeNo > getSizeZ() && populateCore) core.get(0).sizeZ++;
      }
      else if (qName.equals("WellShape")) {
        parentQName = qName;
      }
      else if (qName.equals("Image")) {
        parentQName = qName;
        nextImage++;

        if (level != MetadataLevel.MINIMUM) {
          //Implemented for FLEX v1.7 and below
          String x = attributes.getValue("CameraBinningX");
          String y = attributes.getValue("CameraBinningY");
          if (x != null) binX = Integer.parseInt(x);
          if (y != null) binY = Integer.parseInt(y);
        }
      }
      else if (qName.equals("Plate")) {
        parentQName = qName;
        if (qName.equals("Plate")) {
          nextPlate++;
          plateCount++;
        }
      }
      else if (qName.equals("WellCoordinate")) {
        if (wellNumber.length == 1) {
          wellNumber[0][0] = Integer.parseInt(attributes.getValue("Row")) - 1;
          wellNumber[0][1] = Integer.parseInt(attributes.getValue("Col")) - 1;
        }
      }
      else if (qName.equals("Slider") && level != MetadataLevel.MINIMUM) {
        sliderName = attributes.getValue("Name");
      }
      else if (qName.equals("Filter") && level != MetadataLevel.MINIMUM) {
        String id = attributes.getValue("ID");
        if (sliderName.endsWith("Dichro")) {
          String dichroicID =
            MetadataTools.createLSID("Dichroic", 0, nextDichroic);
          dichroicMap.put(id, dichroicID);
          store.setDichroicID(dichroicID, 0, nextDichroic);
          store.setDichroicModel(id, 0, nextDichroic);
          nextDichroic++;
        }
        else {
          String filterID = MetadataTools.createLSID("Filter", 0, nextFilter);
          filterMap.put(id, filterID);
          store.setFilterID(filterID, 0, nextFilter);
          store.setFilterModel(id, 0, nextFilter);
          store.setFilterFilterWheel(sliderName, 0, nextFilter);
          nextFilter++;
        }
      }
      else if (qName.equals("FilterCombination") &&
        level != MetadataLevel.MINIMUM)
      {
        filterSet = "FilterSet:" + attributes.getValue("ID");
        filterSetMap.put(filterSet, new FilterGroup());
      }
      else if (qName.equals("SliderRef") && level != MetadataLevel.MINIMUM) {
        String filterName = attributes.getValue("Filter");
        String slider = attributes.getValue("ID");
        FilterGroup group = filterSetMap.get(filterSet);
        if (nextSliderRef == 0 && slider.startsWith("Camera")) {
          group.emission = filterMap.get(filterName);
        }
        else if (nextSliderRef == 1 && slider.startsWith("Camera")) {
          group.excitation = filterMap.get(filterName);
        }
        else if (slider.equals("Primary_Dichro")) {
          group.dichroic = dichroicMap.get(filterName);
        }
        String lname = filterName.toLowerCase();
        if (!lname.startsWith("empty") && !lname.startsWith("blocked")) {
          nextSliderRef++;
        }
        filterSetMap.put(filterSet, group);
      }
    }
  }

  /** SAX handler for parsing XML from .mea files. */
  public class MeaHandler extends BaseHandler {
    private Vector<String> flex = new Vector<String>();
    private String[] hostnames = null;

    // -- MeaHandler API methods --

    public Vector<String> getFlexFiles() { return flex; }

    // -- DefaultHandler API methods --

    public void startElement(String uri,
      String localName, String qName, Attributes attributes)
    {
      if (qName.equals("Host")) {
        String hostname = attributes.getValue("name");
        LOGGER.debug("FlexHandler: found hostname '{}'", hostname);
        hostnames = serverMap.get(hostname);
        if (hostnames != null) {
          LOGGER.debug("Sanitizing hostnames...");
          for (int i=0; i<hostnames.length; i++) {
            String host = hostnames[i];
            hostnames[i] = hostnames[i].replace('/', File.separatorChar);
            hostnames[i] = hostnames[i].replace('\\', File.separatorChar);
            LOGGER.debug("Hostname #{} was {}, is now {}",
              new Object[] {i, host, hostnames[i]});
          }
        }
      }
      else if (qName.equals("Picture")) {
        String path = attributes.getValue("path");
        if (!path.endsWith(".flex")) path += ".flex";
        path = path.replace('/', File.separatorChar);
        path = path.replace('\\', File.separatorChar);
        LOGGER.debug("Found .flex in .mea: {}", path);
        if (hostnames != null) {
          int numberOfFlexFiles = flex.size();
          for (String hostname : hostnames) {
            String filename = hostname + File.separator + path;
            if (new Location(filename).exists()) {
              flex.add(filename);
            }
          }
          if (flex.size() == numberOfFlexFiles) {
            LOGGER.warn("{} was in .mea, but does not actually exist.", path);
          }
        }
      }
    }
  }

  /** SAX handler for parsing XML from .res files. */
  public class ResHandler extends BaseHandler {

    // -- DefaultHandler API methods --

    public void startElement(String uri,
      String localName, String qName, Attributes attributes)
    {
      if (qName.equals("AnalysisResults")) {
        plateAcqStartTime = attributes.getValue("date");
      }
    }

  }

  /** Stores a grouping of filters. */
  class FilterGroup {
    public String emission;
    public String excitation;
    public String dichroic;
    public String id;
  }

  // -- FlexReader API methods --

  /**
   * Add the path 'realName' to the mapping for the server named 'alias'.
   * @throws FormatException if 'realName' does not exist
   */
  public static void appendServerMap(String alias, String realName)
    throws FormatException
  {
    LOGGER.debug("appendServerMap({}, {})", alias, realName);

    if (alias != null) {
      if (realName == null) {
        LOGGER.debug("removing mapping for {}", alias);
        serverMap.remove(alias);
      }
      else {
        // verify that 'realName' exists
        Location server = new Location(realName);
        if (!server.exists()) {
          throw new FormatException("Server " + realName + " was not found.");
        }
        String[] names = serverMap.get(alias);
        if (names == null) {
          serverMap.put(alias, new String[] {realName});
        }
        else {
          String[] tmpNames = new String[names.length + 1];
          System.arraycopy(names, 0, tmpNames, 0, names.length);
          tmpNames[tmpNames.length - 1] = realName;
          serverMap.put(alias, tmpNames);
        }
      }
    }
  }

  /**
   * Map the server named 'alias' to the path 'realName'.
   * If other paths were mapped to 'alias', they will be overwritten.
   */
  public static void mapServer(String alias, String realName) {
    LOGGER.debug("mapSever({}, {})", alias, realName);
    if (alias != null) {
      if (realName == null) {
        LOGGER.debug("removing mapping for {}", alias);
        serverMap.remove(alias);
      }
      else {
        LOGGER.debug("Finding base server name...");
        if (realName.endsWith(File.separator)) {
          realName = realName.substring(0, realName.length() - 1);
        }
        String baseName = realName;
        if (baseName.endsWith(SCREENING)) {
          baseName = baseName.substring(0, baseName.lastIndexOf(SCREENING));
        }
        else if (baseName.endsWith(ARCHIVE)) {
          baseName = baseName.substring(0, baseName.lastIndexOf(ARCHIVE));
        }
        LOGGER.debug("Base server name is {}", baseName);

        Vector<String> names = new Vector<String>();
        names.add(realName);
        Location screening =
          new Location(baseName + File.separator + SCREENING);
        Location archive = new Location(baseName + File.separator + ARCHIVE);

        if (screening.exists()) names.add(screening.getAbsolutePath());
        if (archive.exists()) names.add(archive.getAbsolutePath());

        LOGGER.debug("Server names for {}:", alias);
        for (String name : names) {
          LOGGER.debug("  {}", name);
        }

        mapServer(alias, names.toArray(new String[names.size()]));
      }
    }
  }

  /**
   * Map the server named 'alias' to the paths in 'realNames'.
   * If other paths were mapped to 'alias', they will be overwritten.
   */
  public static void mapServer(String alias, String[] realNames) {
    StringBuffer msg = new StringBuffer("mapServer(");
    msg.append(alias);
    if (realNames != null) {
      msg.append(", [");
      for (String name : realNames) {
        msg.append(name);
        msg.append(", ");
      }
      msg.append("])");
    }
    else msg.append(", null)");
    LOGGER.debug(msg.toString());

    if (alias != null) {
      if (realNames == null) {
        LOGGER.debug("Removing mapping for {}", alias);
        serverMap.remove(alias);
      }
      else {
        for (String server : realNames) {
          try {
            appendServerMap(alias, server);
          }
          catch (FormatException e) {
            LOGGER.debug("Failed to map server '{}'", server, e);
          }
        }
      }
    }
  }

  /**
   * Read a configuration file with lines of the form:
   *
   * &lt;server alias&gt;=&lt;real server name&gt;
   *
   * and call mapServer(String, String) accordingly.
   *
   * @throws FormatException if configFile does not exist.
   * @see #mapServer(String, String)
   */
  public static void mapServersFromConfigurationFile(String configFile)
    throws FormatException, IOException
  {
    LOGGER.debug("mapServersFromConfigurationFile({})", configFile);
    Location file = new Location(configFile);
    if (!file.exists()) {
      throw new FormatException(
        "Configuration file " + configFile + " does not exist.");
    }

    String[] lines = DataTools.readFile(configFile).split("[\r\n]");
    for (String line : lines) {
      LOGGER.trace(line);
      int eq = line.indexOf("=");
      if (eq == -1 || line.startsWith("#")) continue;
      String alias = line.substring(0, eq).trim();
      String[] servers = line.substring(eq + 1).trim().split(";");
      mapServer(alias, servers);
    }
  }

}
