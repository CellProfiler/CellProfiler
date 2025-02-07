import io
import pickle

from ._image_set import ImageSet
from ..utilities.image import make_dictionary_key


class ImageSetList:
    """Represents the list of image sets in a pipeline run

    """

    def __init__(self, test_mode=False):
        self.__image_sets = []
        self.__image_sets_by_key = {}
        self.legacy_fields = {}
        self.__associating_by_key = None
        self.combine_path_and_file = False
        self.test_mode = test_mode

    def add_provider_to_all_image_sets(self, provider):
        """Provide an image to every image set

        provider - an instance of AbstractImageProvider
        """
        for image_set in self.__image_sets:
            image_set.add_provider(provider)

    def count(self):
        return len(self.__image_sets)

    def get_groupings(self, keys):
        """Return the groupings of an image set list over a set of keys

        keys - a sequence of keys that match some of the image set keys

        returns an object suitable for use by CPModule.get_groupings:
        tuple of keys, groupings
        keys - the keys as passed into the function
        groupings - a sequence of groupings of image sets where
                    each element of the sequence is a two-tuple.
                    The first element of the two-tuple is a dictionary
                    that gives the group's values for each key.
                    The second element is a list of image numbers of
                    the images in the group
        """
        #
        # Sort order for dictionary keys
        #
        sort_order = []
        dictionaries = []
        #
        # Dictionary of key_values to list of image numbers
        #
        d = {}
        for i in range(self.count()):
            image_set = self.get_image_set(i)
            assert isinstance(image_set, ImageSet)
            key_values = tuple([str(image_set.keys[key]) for key in keys])
            if key_values not in d:
                d[key_values] = []
                sort_order.append(key_values)
            d[key_values].append(i + 1)
        return keys, [(dict(list(zip(keys, k))), d[k]) for k in sort_order]

    def get_image_set(self, keys_or_number):
        """Return either the indexed image set (keys_or_number = index) or the image set with matching keys

        """
        if not isinstance(keys_or_number, dict):
            keys = {"number": keys_or_number}
            number = keys_or_number
            if self.__associating_by_key is None:
                self.__associating_by_key = False
            k = make_dictionary_key(keys)
        else:
            keys = keys_or_number
            k = make_dictionary_key(keys)
            if k in self.__image_sets_by_key:
                number = self.__image_sets_by_key[k].number
            else:
                number = len(self.__image_sets)
            self.__associating_by_key = True
        if number >= len(self.__image_sets):
            self.__image_sets += [None] * (number - len(self.__image_sets) + 1)
        if self.__image_sets[number] is None:
            image_set = ImageSet(number, keys, self.legacy_fields)
            self.__image_sets[number] = image_set
            self.__image_sets_by_key[k] = image_set
            if self.__associating_by_key:
                k = make_dictionary_key(dict(number=number))
                self.__image_sets_by_key[k] = image_set
        else:
            image_set = self.__image_sets[number]
        return image_set

    def load_state(self, state):
        """Load an image_set_list's state from the string returned from save_state"""

        self.__image_sets = []
        self.__image_sets_by_key = {}

        # Make a safe unpickler
        p = pickle.Unpickler(io.BytesIO(state))

        count = p.load()

        all_keys = [p.load() for i in range(count)]

        self.legacy_fields = p.load()

        #
        # Have to do in this order in order for the image set's
        # legacy_fields property to hook to the right legacy_fields
        #
        for i in range(count):
            self.get_image_set(all_keys[i])

    def purge_image_set(self, number):
        """Remove the memory associated with an image set"""
        keys = self.__image_sets[number].keys
        image_set = self.__image_sets[number]
        for provider in image_set.providers:
            provider.release_memory()
        self.__image_sets[number] = None
        self.__image_sets_by_key[repr(keys)] = None

    def save_state(self):
        """Return a string that can be used to load the image_set_list's state

        load_state will restore the image set list's state. No image_set can
        have image providers before this call.
        """
        f = io.BytesIO()
        pickle.dump(self.count(), f)
        for i in range(self.count()):
            image_set = self.get_image_set(i)
            assert isinstance(image_set, ImageSet)
            assert (
                len(image_set.providers) == 0
            ), "An image set cannot have providers while saving its state"
            pickle.dump(image_set.keys, f)
        pickle.dump(self.legacy_fields, f)
        return f.getvalue()
