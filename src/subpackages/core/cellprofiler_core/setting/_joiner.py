from ._setting import Setting
from ._validation_error import ValidationError


class Joiner(Setting):
    """The joiner setting defines a joining condition between conceptual tables

    You might want to join several tables by specifying the columns that match
    each other or might want to join images in an image set by matching
    their metadata. The joiner takes a dictionary of lists of column names
    or metadata keys where the dictionary key holds the table or image name
    and the list of values holds the names of table columns or metadata keys.

    The joiner's value is, conceptually, a list of dictionaries where each
    dictionary in the list documents how to join one column or metadata key
    in one of the tables or images to the others.

    The conceptual value is a list of dictionaries of unicode string keys
    and values (or value = None). This can be encoded using str() and
    can be decoded using eval.
    """

    def __init__(self, text, value="[]", allow_none=True, **kwargs):
        """Initialize the joiner

        text - label to the left of the joiner

        value - "repr" done on the joiner's underlying structure which is
                a list of dictionaries

        allow_none - True (by default) to allow one of the entities to have
                     None for a join, indicating that it matches against
                     everything
        """
        super(self.__class__, self).__init__(text, value, **kwargs)
        self.entities = {}
        self.allow_none = allow_none

    def parse(self):
        """Parse the value into a list of dictionaries

        return a list of dictionaries where the key is the table or image name
        and the value is the column or metadata
        """
        return eval(self.value_text, {"__builtins__": None}, {})

    def default(self):
        """Concoct a default join as a guess if setting is uninitialized"""
        all_names = {}
        best_name = None
        best_count = 0
        for value_list in list(self.entities.values()):
            for value in value_list:
                if value in all_names:
                    all_names[value] += 1
                else:
                    all_names[value] = 1
                if best_count < all_names[value]:
                    best_count = all_names[value]
                    best_name = value
        if best_count == 0:
            return []
        else:
            return [
                dict(
                    [
                        (k, best_name if best_name in self.entities[k] else None)
                        for k in list(self.entities.keys())
                    ]
                )
            ]

    def build(self, dictionary_list):
        """Build a value from a list of dictionaries"""
        self.value = self.build_string(dictionary_list)

    @classmethod
    def build_string(cls, dictionary_list):
        return str(dictionary_list)

    def test_valid(self, pipeline):
        """Test the joiner setting to ensure that the join is supported

        """
        join = self.parse()
        if len(join) == 0:
            raise ValidationError(
                "This setting needs to be initialized by choosing items from each column",
                self,
            )
        for d in join:
            for column_name, value in list(d.items()):
                if column_name in self.entities and (
                    value not in self.entities[column_name] and value is not None
                ):
                    raise ValidationError(
                        "%s is not a valid choice for %s" % (value, column_name), self
                    )
