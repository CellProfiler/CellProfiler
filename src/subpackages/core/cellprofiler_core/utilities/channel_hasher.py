import collections

from cellprofiler_core.pipeline import ImagePlane


class ChannelHasher:
    """
    This class is used by NamesAndTypes to pair image sets by metadata.
    It forms a wrapper around a defaultdict object which returns lists.
    Those lists will store ImagePlane objects matching a set of
    metadata values.

    channel_name is the name of the image channel.

    keymap is a list of metadata keys to be used to index the ImagePlane objects.

    An ImagePlane is assigned a hash key tuple based on the keymap parameter.
    Sometimes a channel doesn't use all metadata keys, in which case the unused
    key should be passed as None within the keymap. All ChannelHasher objects
    in use should have a key list of the same length.

    e.g. If we're using Plate, Well and Site keys to match metadata,
    ['Plate', 'Well', 'Site'] should be the keymap for the channel. When adding
    planes to the channel group they'll be indexed with a tuple of (Plate, Well, Site)
    values. For successful image set creation each hash key should only end up with a
    single ImagePlane.

    If we have an Illum function which is assigned based on just the Plate key, the
    keymap should be ['Plate', None, None]. Internally the hasher will call
    for just (Plate) when NamesAndTypes calls for Plate X, Well Y, Site Z. This
    allows a single image to match in multiple sets.
    """
    def __init__(self, channel_name, keymap):
        self.name = channel_name
        self.mapper = collections.defaultdict(list)
        self.keymap = keymap
        self.hash_keys = [key for key in keymap if key is not None]

    def add(self, image_plane: ImagePlane):
        plane_hash = tuple([image_plane.get_metadata(key) for key in self.hash_keys])
        self.mapper[plane_hash].append(image_plane)

    def keys(self):
        return self.mapper.keys()

    def __getitem__(self, key):
        real_key = []
        for key, mapped_key in zip(key, self.keymap):
            if mapped_key is None:
                continue
            real_key.append(key)
        return self.mapper[tuple(real_key)]

    def __setitem__(self, key, value):
        if len(key) == len(self.hash_keys):
            self.mapper[key].append(value)
        else:
            real_key = []
            for key, mapped_key in zip(key, self.keymap):
                if mapped_key is None:
                    continue
                real_key.append(key)
            self.mapper[tuple(real_key)] = value
