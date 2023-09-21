"""<b>OmeroLoadImages</b> loads one or more images from <i>OMERO</i>.
<hr>

This module retrieves all images in a dataset or plate from an OMERO server.
It is also possible to load a single image (e.g., for testing your pipeline).

<b>Important note</b>

<p>In OMERO, images contain <i>image planes</i>. It is these image planes that are
considered images in their own right in Cellprofiler. When this module refers to an image,
it will mean a Cellprofiler image (= OMERO image plane) unless noted otherwise.</p>

<b>Running in headless mode</b>

<p>When run from the command line the module will not used the omero object id that is saved in
the pipeline. It will use the image directory parameter instead. e.g.:</p>

<i>cellprofiler -p mypipeline -i 1</i>

In the above example the pipeline "mypipeline" will be run with "1" as omero object id.

<h2>OMERO</h2>

<ul>
<li>OpenMicroscopy Environment (OME)</li>
<li>www.openmicroscopy.org.uk</li>
<li>University of Dundee</li>
</ul>
"""


# module author: Bram Gerritsen
# e-mail: b.gerritsen@nki.nl

import traceback

import numpy as np
import wx

import cellprofiler_core.image as cpimage
import cellprofiler_core.module as cpm
import cellprofiler_core.measurement as cpmeas
import cellprofiler_core.preferences as cpp
import cellprofiler_core.setting as cps

# get the default cellprofiler image names for the different
# channels of an omero image from the loadimages module
from cellprofiler_core.modules import default_cpimage_name

import omero
from omero.rtypes import rlong
from omero.rtypes import rint
from omero_version import omero_version

# omero beta 4 versions that did not fully support High Content Screening (HCS)
OMERO_VERSION4_PREHCS = ["Beta-4.0.1", "Beta-4.0.2"]

# omero beta 4.0.3 also requires the following import,
# but that version identified itself with 'Beta-4.0.2'
if omero_version in OMERO_VERSION4_PREHCS:
    # This import is required because of an issue with forward-declarations in Ice
    import omero_api_Gateway_ice

    DEFAULT_OMERO_PORT = 4063
    INT_8 = "int8"
    UINT_8 = "uint8"
    INT_16 = "int16"
    UINT_16 = "uint16"
    INT_32 = "int32"
    UINT_32 = "uint32"
    FLOAT = "float"
    DOUBLE = "double"
else:
    DEFAULT_OMERO_PORT = 4064
    from omero.util.pixelstypetopython import *

# strings for choice variables
MS_IMAGE = "Image"
MS_DATASET = "Dataset"
MS_PLATE = "Plate"

#
# Defaults for the module settings
#
DEFAULT_OMERO_HOST = "localhost"
DEFAULT_OMERO_USERNAME = ""
DEFAULT_OMERO_PASSWORD = ""
DEFAULT_OMERO_OBJECT = MS_IMAGE
DEFAULT_OMERO_OBJECT_ID = 1

# The different categories that this module
# provides measurements for.
""" The Dataset measurement category"""
C_DATASET = "OmeroDataset"

""" The Plate measurement category"""
C_PLATE = "OmeroPlate"

""" The Well measurement category"""
C_WELL = "OmeroWell"

""" The Image measurement category"""
C_IMAGE = "OmeroImage"

""" The Pixels measurement category"""
C_PIXELS = "OmeroPixels"

# Features
FTR_NAME = "Name"
FTR_ID = "Id"
FTR_ROW = "Row"
FTR_COLUMN = "Column"
FTR_Z = "Z"
FTR_C = "C"
FTR_T = "T"

"""The Dataset name measurement name"""
M_DATASET_NAME = "%s_%s" % (C_DATASET, FTR_NAME)

"""The Dataset id measurement name"""
M_DATASET_ID = "%s_%s" % (C_DATASET, FTR_ID)

"""The  Plate name measurement name"""
M_PLATE_NAME = "%s_%s" % (C_PLATE, FTR_NAME)

"""The Plate id measurement name"""
M_PLATE_ID = "%s_%s" % (C_PLATE, FTR_ID)

"""The Well row measurement name"""
M_WELL_ROW = "%s_%s" % (C_WELL, FTR_ROW)

"""The Well column measurement name"""
M_WELL_COLUMN = "%s_%s" % (C_WELL, FTR_COLUMN)

"""The Well id measurement name"""
M_WELL_ID = "%s_%s" % (C_WELL, FTR_ID)

"""The Image name measurement name (note: name of the image in omero)"""
M_IMAGE_NAME = "%s_%s" % (C_IMAGE, FTR_NAME)

"""The Image id measurement name (note: image id of the image in omero)"""
M_IMAGE_ID = "%s_%s" % (C_IMAGE, FTR_ID)

"""The Pixels id measurement name """
M_PIXELS_ID = "%s_%s" % (C_PIXELS, FTR_ID)

"""The channel number """
M_C = "%s_%s" % (cpmeas.C_METADATA, FTR_C)

"""The Z depth measurement name """
M_Z = "%s_%s" % (cpmeas.C_METADATA, FTR_Z)

"""The Time index measurement name """
M_T = "%s_%s" % (cpmeas.C_METADATA, FTR_T)

"""The provider name for the omero image provider"""
P_OMERO = "OmeroImageProvider"
"""The version number for the __init__ method of the omero image provider"""
V_OMERO = 1


def create_omero_gateway(
    host=DEFAULT_OMERO_HOST,
    port=DEFAULT_OMERO_PORT,
    username=DEFAULT_OMERO_USERNAME,
    password=DEFAULT_OMERO_PASSWORD,
):
    """Connect to an omero server and create an omero gateway instance"""
    try:
        omero_client = omero.client(host, port)
        omero_session = omero_client.createSession(username, password)
        omero_gateway = omero_session.createGateway()
    except Exception as err:
        raise RuntimeError(
            "Unable to connect to OMERO server %s@%s:%d" % (username, host, int(port)),
            err,
        )
    return omero_client, omero_session, omero_gateway


class OmeroLoadImages(cpm.Module):
    variable_revision_number = 1
    module_name = "OmeroLoadImages"
    category = "File Processing"

    # Make the omero client object an attribute of this class, because otherwise the
    # Ice Communicator will be disconnected when the omero client goes out
    # of scope and is cleaned up by python's garbage collector
    omero_client = None
    omero_session = None
    omero_gateway = None

    def create_settings(self):
        self.omero_host = cps.Text(
            "Host address",
            DEFAULT_OMERO_HOST,
            doc="""Host address of an omero server. Can be an ip-address or a hostname.""",
        )
        self.omero_port = cps.Integer(
            "Port", DEFAULT_OMERO_PORT, doc="""Port of an omero server."""
        )
        self.omero_username = cps.Text(
            "Username",
            DEFAULT_OMERO_USERNAME,
            doc="""Username is required for login into an omero server.""",
        )
        self.omero_password = cps.Text(
            "Password",
            DEFAULT_OMERO_PASSWORD,
            doc="""Password is required for login into an omero server.""",
        )
        self.omero_object = cps.Choice(
            "Object to load", [MS_IMAGE, MS_DATASET, MS_PLATE], DEFAULT_OMERO_OBJECT
        )
        self.omero_object_id = cps.Integer(
            "Object id",
            DEFAULT_OMERO_OBJECT_ID,
            doc="""This is a number that omero uses to uniquely identify an object, be it a dataset, plate, or image.""",
        )
        self.load_channels = cps.DoSomething(
            "", "Load channels from OMERO", self.load_channels
        )

        # All the omero images that are loaded are assumed to have
        # as many or more channels than the highest channel number
        # the user specifies.
        self.channels = []
        self.channel_count = cps.HiddenCount(self.channels, "Channel count")
        # Add the first channel
        self.add_channelfn(False)

        # Button for adding other channels
        self.add_channel = cps.DoSomething(
            "", "Add another channel", self.add_channelfn
        )

    def create_omero_gateway(self):
        """Create omero gateway based on module settings """
        if self.omero_client is not None:
            self.omero_client.closeSession()
        self.omero_client, self.omero_session, self.omero_gateway = create_omero_gateway(
            self.omero_host.value,
            self.omero_port.value,
            self.omero_username.value,
            self.omero_password.value,
        )

    def get_omero_plate(self, plate_id):
        """Get plate from omero

        id - id of plate in omero
        """
        return omero_session.getQueryService().findByString("Plate", "id", plate_id)

    def load_channels(self):
        """Add and set channels based on an image from omero """
        try:
            self.create_omero_gateway()
            id = int(self.omero_object_id.value)

            if self.omero_object == MS_IMAGE:
                omero_image = self.omero_gateway.getImage(id)
            elif self.omero_object == MS_DATASET:
                images_from_dataset = self.get_images_from_dataset(id, 1)
                if len(images_from_dataset) == 0:
                    omero_image = None
                else:
                    omero_image = images_from_dataset[0]
            elif self.omero_object == MS_PLATE:
                wells_from_plate = self.get_wells_from_plate(id, 1)
                if len(wells_from_plate) == 0:
                    omero_image = None
                else:
                    omero_well = wells_from_plate[0]
                    omero_image = omero_well.getWellSample(0).getImage()

            if omero_image is None:
                # Don't say plate or dataset not found as they might still exist but do not have
                # images attached to them. Another reason for not being able to find images is
                # because the omero account used does not have permissions to retrieve the image
                # or images.
                raise RuntimeError(
                    "No image found for %s with id %d" % (self.omero_object, id)
                )

            omero_image_id = omero_image.getId().getValue()
            pixels = self.omero_gateway.getPixelsFromImage(omero_image_id)[0]
            # The pixels doesn't have all the (logical) channels
            # because of lazy loading. So the pixels is requested
            # again, but in a way to get the channels as well.
            pixels = self.omero_gateway.getPixels(pixels.getId().getValue())

            # repopulate channels based on the retrieved pixels.
            # note: cannot say self.channels=[] because the channel_count
            # is associated with the object self.channels refers to.
            for channel in self.channels[:]:
                self.channels.remove(channel)
            omero_channels = [channel for channel in pixels.iterateChannels()]
            number_of_channels = pixels.getSizeC().getValue()
            for channel_number in range(0, number_of_channels):
                omero_channel = omero_channels[channel_number].getLogicalChannel()
                # load default cpimage name in case the logical channel name
                # cannot be retrieved. e.g., when the logical channel name is null
                try:
                    omero_channel_name = omero_channel.getName().getValue().strip()
                except:
                    omero_channel_name = default_cpimage_name(channel_number)
                self.add_channelfn(channel_number != 0)
                self.channels[-1].cpimage_name.set_value(omero_channel_name)
                self.channels[-1].channel_number.set_value(str(channel_number))

            # Close the session just in case the user decides not to run
            # the pipeline.
            self.omero_client.closeSession()
            self.omero_client = None
            wx.MessageBox(
                "Retrieved %d channel(s) from OMERO" % number_of_channels,
                "",
                wx.ICON_INFORMATION,
            )
        except:
            wx.MessageBox(traceback.format_exc(limit=0), "Exception", wx.ICON_ERROR)

    def add_channelfn(self, can_remove=True):
        """Add another image channel

        can_remove - true if we are allowed to remove this channel
        """
        group = cps.SettingsGroup()
        self.channels.append(group)

        # Check which cellprofiler image we are in the group
        # (each channel translates to a single cellprofiler image)
        cpimg_index = 0
        for channel in self.channels:
            if id(channel) == id(group):
                break
            cpimg_index += 1

        group.append("divider", cps.Divider(line=True))
        group.append(
            "cpimage_name",
            cps.ImageNameProvider("Image name", default_cpimage_name(cpimg_index)),
        )
        channel_numbers = [str(x) for x in range(0, max(10, len(self.channels) + 2))]
        group.append(
            "channel_number",
            cps.Choice(
                "Channel number:",
                channel_numbers,
                channel_numbers[len(self.channels) - 1],
                doc="""(Used only for multichannel images)
			The channels of a multichannel image are numbered starting from 0 (zero).

			Each channel is a greyscale image, acquired using different
			illumination sources and/or optics. Use this setting to pick
			the channel to associate with the image or images you load from
			OMERO.""",
            ),
        )
        group.can_remove = can_remove
        if can_remove:
            group.append(
                "remover",
                cps.RemoveSettingButton(
                    "Remove this channel", "Remove channel", self.channels, group
                ),
            )

    def settings(self):
        varlist = [
            self.omero_host,
            self.omero_port,
            self.omero_username,
            self.omero_password,
            self.omero_object,
            self.omero_object_id,
            self.channel_count,
        ]
        for channel in self.channels:
            varlist += [channel.cpimage_name, channel.channel_number]
        return varlist

    def visible_settings(self):
        varlist = [
            self.omero_host,
            self.omero_port,
            self.omero_username,
            self.omero_password,
            self.omero_object,
            self.omero_object_id,
            self.load_channels,
        ]
        for channel in self.channels:
            varlist += [channel.divider, channel.cpimage_name, channel.channel_number]
            if channel.can_remove:
                varlist += [channel.remover]
        varlist.append(self.add_channel)
        return varlist

    def validate_module(self, pipeline):
        """Validate a module's settings"""
        for setting in (
            self.omero_host,
            self.omero_port,
            self.omero_username,
            self.omero_password,
        ):
            if setting.value == "":
                raise cps.ValidationError(
                    'Cannot continue if "%s" is not set.' % setting.get_text(), setting
                )

    def is_load_module(self):
        """This module creates image sets so it is a load module"""
        return True

    def prepare_run(self, workspace):
        """Set up omero image providers inside the image_set_list"""
        pipeline = workspace.pipeline
        image_set_list = workspace.image_set_list
        if pipeline.in_batch_mode():
            # TODO: Rewrite the OmeroImageProvider such that it can be used in batch mode
            # e.g., omero session keys could be used to attach to existing sessions to
            # keep OmeroImageProviders from creating a new session every time an image should be loaded
            return False

        if cpp.get_headless():
            print(
                "OmeroLoadImages running in headless mode: image directory parameter will be used as omero object id"
            )
            self.omero_object_id.set_value(int(cpp.get_default_image_directory()))
            print("omero object id = %d" % self.omero_object_id.value)
            print("omero object type = %s" % self.omero_object.value)

        self.create_omero_gateway()
        if self.omero_object == MS_IMAGE:
            omero_image_list = [self.omero_gateway.getImage(self.omero_object_id.value)]
        elif self.omero_object == MS_DATASET:
            # Get dataset without leaves(=images&pixels)
            dataset = self.omero_gateway.getDataset(self.omero_object_id.value, False)
            self.dataset_name = dataset.getName().getValue()
            omero_image_list = self.get_images_from_dataset(self.omero_object_id.value)
        elif self.omero_object == MS_PLATE:
            self.wells = self.get_wells_from_plate(self.omero_object_id.value)
            self.plate_name = self.wells[0].getPlate().getName().getValue()
            omero_image_list = []
            for well in self.wells:
                for wellsample in well.iterateWellSamples():
                    omero_image_list.append(wellsample.getImage())

        # get names and pixels from omero images
        pixels_list = []
        for omero_image in omero_image_list:
            image_id = omero_image.getId().getValue()
            pixels_list += self.omero_gateway.getPixelsFromImage(image_id)

        # add images to image sets
        image_set_count = len(pixels_list)
        for i in range(0, image_set_count):
            image_set = image_set_list.get_image_set(i)
            pixels = pixels_list[i]
            pixels_id = pixels.getId().getValue()
            sizeZ = pixels.getSizeZ().getValue()
            sizeC = pixels.getSizeC().getValue()
            sizeT = pixels.getSizeT().getValue()
            for channel in self.channels:
                for z in range(0, sizeZ):
                    for t in range(0, sizeT):
                        c = int(channel.channel_number.value)
                        self.save_image_set_info(
                            image_set,
                            channel.cpimage_name.value,
                            P_OMERO,
                            V_OMERO,
                            self.omero_gateway,
                            pixels_id,
                            z,
                            c,
                            t,
                        )
        return True

    def get_images_from_dataset(self, dataset_id, limit=None):
        """Get images from dataset
        limit - maximum number of images to retrieve
        """
        q = self.omero_session.getQueryService()
        p = omero.sys.Parameters()
        p.map = {}
        p.map["oid"] = rlong(int(dataset_id))
        if limit is not None:
            f = omero.sys.Filter()
            f.limit = rint(int(limit))
            p.theFilter = f
        sql = (
            "select im from Image im "
            "left outer join fetch im.datasetLinks dil left outer join fetch dil.parent d "
            "where d.id = :oid order by im.id asc"
        )
        return q.findAllByQuery(sql, p)

    def get_wells_from_plate(self, plate_id, limit=None):
        """ Retrieves every well of a plate that has an image attached to it
        (via a wellsample of course).

            plate_id	- id of the plate
            limit		- maximum number of wells to retrieve
        """
        q = self.omero_session.getQueryService()
        p = omero.sys.Parameters()
        p.map = {}
        p.map["oid"] = rlong(int(plate_id))
        if limit is not None:
            f = omero.sys.Filter()
            f.limit = rint(int(limit))
            p.theFilter = f
        sql = (
            "select well from Well as well "
            "left outer join fetch well.plate as pt "
            "left outer join fetch well.wellSamples as ws "
            "inner join fetch ws.image as img "
            "where well.plate.id = :oid"
        )
        return q.findAllByQuery(sql, p)

    def save_image_set_info(self, image_set, image_name, provider, version, *args):
        """Write out the details for creating an image provider

        Write information to the image set list legacy fields for saving
        the state needed to create an image provider.

        image_set - create a provider on this image set
        image_name - the image name for the image
        provider - the name of an image set provider (the name will be read
                by load_image_set_info to create the actual provider)
        version - the version # of the provider, in case the arguments change
        args - string arguments that will be passed to the provider's init fn
        """
        if provider != P_OMERO:
            raise NotImplementedError(
                "provider %s has not been implemented by this module" % provider
            )
        d = self.get_dictionary(image_set)
        d[image_name] = [provider, version] + list(args)

    def load_image_set_info(self, image_set):
        """Loads the image set information, creating the providers"""
        d = self.get_dictionary(image_set)
        for image_name in list(d.keys()):
            values = d[image_name]
            provider, version = values[:2]
            if (provider, version) == (P_OMERO, V_OMERO):
                omero_gateway, pixels_id, z, c, t = values[2:]
                image_set.providers.append(
                    OmeroImageProvider(image_name, omero_gateway, pixels_id, z, c, t)
                )
            else:
                raise NotImplementedError(
                    "Can't restore file information: image provider %s and/or version %d not supported"
                    % provider,
                    version,
                )

    def get_dictionary(self, image_set):
        """Get the module's legacy fields dictionary for this image set"""
        key = "%s:%d" % (self.module_name, self.module_num)
        if key not in image_set.legacy_fields:
            image_set.legacy_fields[key] = {}
        d = image_set.legacy_fields[key]
        if image_set.image_number not in d:
            d[image_set.image_number] = {}
        return d[image_set.image_number]

    def prepare_group(self, workspace, grouping, image_numbers):
        """Load the images from the dictionary into the image sets here"""
        for image_number in image_numbers:
            image_set = workspace.image_set_list.get_image_set(image_number - 1)
            self.load_image_set_info(image_set)

    def run(self, workspace):
        """Run the module. Add the measurements. """

        statistics_dict = {}
        ratio_dict = {}
        for channel in self.channels:
            provider = workspace.image_set.get_image_provider(
                channel.cpimage_name.value
            )
            assert isinstance(provider, OmeroImageProvider)

            name = provider.get_name()
            omero_image_name = provider.get_omero_image_name()
            omero_image_id = provider.get_image_id()
            pixels_id = provider.get_pixels_id()
            z = provider.get_z()
            c = provider.get_c()
            t = provider.get_t()

            header = []
            row = []
            ratio = []
            m = workspace.measurements
            measurements = ()
            if self.omero_object == MS_DATASET:
                measurements += (
                    (M_DATASET_NAME, self.dataset_name, 3.0),
                    (M_DATASET_ID, self.omero_object_id.value, 1.0),
                )
            elif self.omero_object == MS_PLATE:
                # CellProfiler starts counting image sets from 1
                well = self.wells[workspace.measurements.image_set_number - 1]
                well_row = well.getRow().getValue()
                well_column = well.getColumn().getValue()
                well_id = well.getId().getValue()
                measurements += (
                    (M_PLATE_NAME, self.plate_name, 3.0),
                    (M_PLATE_ID, self.omero_object_id.value, 1.0),
                    (M_WELL_ROW, well_row, 1.0),
                    (M_WELL_COLUMN, well_column, 1.0),
                    (M_WELL_ID, well_id, 3.0),
                )
            measurements += (
                (M_IMAGE_NAME, omero_image_name, 3.0),
                (M_IMAGE_ID, omero_image_id, 1.0),
                (M_PIXELS_ID, pixels_id, 1.0),
                (M_Z, z, 0.5),
                (M_C, c, 0.5),
                (M_T, t, 0.5),
            )
            for tag, value, r in measurements:
                m.add_image_measurement("_".join((tag, name)), value)
                header.append(tag)
                row.append(value)
                ratio.append(r)
            statistics = [header, row]
            ratio = [x / sum(ratio) for x in ratio]
            statistics_dict[channel.channel_number.value] = statistics
            ratio_dict[channel.channel_number.value] = ratio

        workspace.display_data.statistics = statistics_dict
        workspace.display_data.ratio = ratio_dict

        if cpp.get_headless():  # headless mode
            for channel in self.channels:
                image_name, channel_number = (
                    channel.cpimage_name.value,
                    channel.channel_number.value,
                )
                print("--- image name: %s\tchannel: %s" % (image_name, channel_number))
                (header, row) = workspace.display_data.statistics[channel_number]
                for i in range(0, len(header)):
                    print("\t%s: %s" % (header[i], row[i]))

    def post_run(self, workspace):
        """Disconnect from the omero server after the run completes"""
        self.omero_client.closeSession()

    def display(self, workspace):
        if workspace.frame is not None:
            figure = workspace.create_or_find_figure(
                title="OmeroLoadImages, image cycle #%d"
                % (workspace.measurements.image_set_number),
                subplots=(2, self.channel_count.value),
            )

            for channel in self.channels:
                image_name, channel_number = (
                    channel.cpimage_name.value,
                    channel.channel_number.value,
                )
                image_set = workspace.image_set
                i, j = 0, int(channel_number)
                pixel_data = image_set.get_image(image_name).pixel_data
                if pixel_data.ndim == 2:
                    figure.subplot_imshow_grayscale(
                        i,
                        j,
                        pixel_data,
                        title=image_name,
                        vmin=0,
                        vmax=1,
                        sharex=figure.subplot(0, 0),
                        sharey=figure.subplot(0, 0),
                    )
                figure.subplot_table(
                    1,
                    int(channel_number),
                    workspace.display_data.statistics[channel_number],
                )

    def get_categories(self, pipeline, object_name):
        """Return the categories that this module produces"""
        if object_name == cpmeas.IMAGE:
            return [C_IMAGE, C_PIXELS, C_METADATA]
        return []

    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces"""
        if object_name == cpmeas.IMAGE:
            return [
                meas.split("_", 1)[1]
                for ob, meas, dtype in self.get_measurement_columns(pipeline)
                if meas.split("_", 1)[0] == category
            ]
        return []

    def get_measurement_columns(self, pipeline):
        """Return a sequence describing the measurement columns needed by this module"""
        cols = []
        for channel in self.channels:
            name = channel.cpimage_name.value
            cols += [
                (
                    cpmeas.IMAGE,
                    "_".join((M_IMAGE_NAME, name)),
                    cpmeas.COLTYPE_VARCHAR_FORMAT % 255,
                ),
                (cpmeas.IMAGE, "_".join((M_IMAGE_ID, name)), cpmeas.COLTYPE_INTEGER),
                (cpmeas.IMAGE, "_".join((M_PIXELS_ID, name)), cpmeas.COLTYPE_INTEGER),
                (cpmeas.IMAGE, "_".join((M_Z, name)), cpmeas.COLTYPE_INTEGER),
                (cpmeas.IMAGE, "_".join((M_C, name)), cpmeas.COLTYPE_INTEGER),
                (cpmeas.IMAGE, "_".join((M_T, name)), cpmeas.COLTYPE_INTEGER),
            ]
            if self.omero_object == MS_DATASET:
                cols += [
                    (
                        cpmeas.IMAGE,
                        "_".join((M_DATASET_NAME, name)),
                        cpmeas.COLTYPE_VARCHAR_FORMAT % 255,
                    ),
                    (
                        cpmeas.IMAGE,
                        "_".join((M_DATASET_ID, name)),
                        cpmeas.COLTYPE_INTEGER,
                    ),
                ]
            elif self.omero_object == MS_PLATE:
                cols += [
                    (
                        cpmeas.IMAGE,
                        "_".join((M_PLATE_NAME, name)),
                        cpmeas.COLTYPE_VARCHAR_FORMAT % 255,
                    ),
                    (
                        cpmeas.IMAGE,
                        "_".join((M_PLATE_ID, name)),
                        cpmeas.COLTYPE_INTEGER,
                    ),
                    (
                        cpmeas.IMAGE,
                        "_".join((M_WELL_ROW, name)),
                        cpmeas.COLTYPE_INTEGER,
                    ),
                    (
                        cpmeas.IMAGE,
                        "_".join((M_WELL_COLUMN, name)),
                        cpmeas.COLTYPE_INTEGER,
                    ),
                    (cpmeas.IMAGE, "_".join((M_WELL_ID, name)), cpmeas.COLTYPE_INTEGER),
                ]
        return cols

    def change_causes_prepare_run(self, setting):
        """Check to see if changing the given setting means you have to restart"""
        # It's safest to say that any change in OmeroLoadImages requires a restart
        return True


# TODO: add exception handling
# TODO: reconnect when gateway has been disconnected?
class OmeroImageProvider(cpimage.AbstractImageProvider):
    """Provide a single image based on omero pixels id"""

    def __init__(self, name, gateway, pixels_id, z=0, c=0, t=0):
        """Initializer

        name		- name of image to be provided
        gateway		- provides connection to an omero server
        pixels_id	- image id relating to the pixels
        z			- z depth
        c			- channel
        t			- time index
        """
        self.__name = name
        self.__gateway = gateway
        self.__pixels_id = int(pixels_id)
        self.__z = int(z)
        self.__c = int(c)
        self.__t = int(t)

        self.__is_cached = False
        self.__cpimage_data = None
        self.__pixels = gateway.getPixels(pixels_id)
        self.__image_id = self.__pixels.getImage().getId().getValue()
        self.__omero_image_name = (
            self.__gateway.getImage(self.__image_id).getName().getValue()
        )

    def provide_image(self, image_set):
        """load an image plane from an omero server
        and return a 2-d grayscale image
        """
        # TODO: return 3d RGB images when c == None like loadimage.py does?

        if self.__is_cached:
            return self.__omero_image_plane

        gateway = self.__gateway
        pixels_id = self.__pixels_id
        z = self.__z
        c = self.__c
        t = self.__t

        # Retrieve the image data from the omero server
        pixels = self.__pixels
        omero_image_plane = gateway.getPlane(pixels_id, z, c, t)

        # Create a 'cellprofiler' image
        width = pixels.getSizeX().getValue()
        height = pixels.getSizeY().getValue()
        pixels_type = pixels.getPixelsType().getValue().getValue()

        # OMERO stores images in big endian format
        little_endian = False
        if pixels_type == INT_8:
            dtype = np.char
            scale = 255
        elif pixels_type == UINT_8:
            dtype = np.uint8
            scale = 255
        elif pixels_type == UINT_16:
            dtype = "<u2" if little_endian else ">u2"
            scale = 65535
        elif pixels_type == INT_16:
            dtype = "<i2" if little_endian else ">i2"
            scale = 65535
        elif pixels_type == UINT_32:
            dtype = "<u4" if little_endian else ">u4"
            scale = 2 ** 32
        elif pixels_type == INT_32:
            dtype = "<i4" if little_endian else ">i4"
            scale = 2 ** 32 - 1
        elif pixels_type == FLOAT:
            dtype = "<f4" if little_endian else ">f4"
            scale = 1
        elif pixels_type == DOUBLE:
            dtype = "<f8" if little_endian else ">f8"
            scale = 1
        else:
            raise NotImplementedError(
                "omero pixels type not implemented for %s" % pixels_type
            )
        # TODO: should something be done here with MaxSampleValue (like loadimages.py does)?

        image = np.frombuffer(omero_image_plane, dtype)
        image.shape = (height, width)
        image = image.astype(np.float32) / float(scale)
        image = cpimage.Image(image)
        self.__cpimage_data = image
        self.__is_cached = True
        return image

    def get_name(self):
        """get the name of this image"""
        return self.__name

    def get_gateway(self):
        """get omero gateway"""
        return self.__gateway

    def get_pixels_id(self):
        """get the pixels id"""
        return self.__pixels_id

    def get_z(self):
        """get the z depth"""
        return self.__z

    def get_c(self):
        """get the channel"""
        return self.__c

    def get_t(self):
        """get the time index"""
        return self.__t

    def get_image_id(self):
        """return the image id (note: id of the image in omero)"""
        return self.__image_id

    def get_omero_image_name(self):
        """return the omero image name of this image"""
        return self.__omero_image_name

    def release_memory(self):
        """Release whatever memory is associated with the image"""
        self.__cpimage_data = None
        self.__is_cached = False
