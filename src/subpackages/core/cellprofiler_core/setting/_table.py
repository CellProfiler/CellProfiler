import cellprofiler_core.utilities.legacy
from ._setting import Setting


class Table(Setting):
    """The Table setting displays a table of values"""

    ATTR_ERROR = "Error"

    def __init__(
        self,
        text,
        min_size=(400, 300),
        max_field_size=30,
        use_sash=False,
        corner_button=None,
        **kwargs,
    ):
        """Constructor

        text - text label to display to the left of the table
        min_size - initial size of the table before user stretches it
        max_field_size - any field with more than this # of characters will
                         be truncated using an ellipsis.
        use_sash - if True, place the table in the bottom resizable sash.
                   if False, place the table inline
        corner_button - if defined, consists of keyword arguments for the corner
                        button mixin: dict(fn_clicked=<function>, label=<label>,
                        tooltip=<tooltip>)
        """
        super(self.__class__, self).__init__(text, "", **kwargs)
        self.column_names = []
        self.data = []
        self.row_attributes = {}
        self.cell_attributes = {}
        self.min_size = min_size
        self.max_field_size = max_field_size
        self.use_sash = use_sash
        self.corner_button = corner_button

    def insert_column(self, index, column_name):
        """Insert a column at the given index

        index - the zero-based index of the column's position

        column_name - the name of the column

        Adds the column to the table and sets the value for any existing
        rows to None.
        """
        self.column_names.insert(index, column_name)
        for row in self.data:
            row.insert(index, None)

    def add_rows(self, columns, data):
        """Add rows to the table

        columns - define the columns for each row of data

        data - rows of data to add. Each field in a row is placed
               at the column indicated by "columns"
        """
        indices = [
            columns.index(c) if c in columns else None for c in self.column_names
        ]
        for row in data:
            self.data.append(
                [None if index is None else row[index] for index in indices]
            )

    def sort_rows(self, columns):
        """Sort rows based on values in columns"""
        indices = [self.column_names.index(c) for c in columns]

        def compare_fn(row1, row2):
            for index in indices:
                x = cellprofiler_core.utilities.legacy.cmp(row1[index], row2[index])
                if x != 0:
                    return x
            return 0

        self.data.sort(compare_fn)

    def clear_rows(self):
        self.data = []
        self.row_attributes = {}
        self.cell_attributes = {}

    def clear_columns(self):
        self.column_names = []

    def get_data(self, row_index, columns):
        """Get the column values for a given row or rows

        row_index - can either be the index of one row or can be a slice or list
                    of rows

        columns - the names of the columns to fetch, in the order they will
                  appear in the row
        """
        column_indices = [self.column_names.index(c) for c in columns]
        if isinstance(row_index, int):
            row_index = slice(row_index, row_index + 1)
        return [[row[ci] for ci in column_indices] for row in self.data[row_index]]

    def set_row_attribute(self, row_index, attribute, set_attribute=True):
        """Set an attribute on a row

        row_index - index of row in question

        attribute - one of the ATTR_ values, for instance ATTR_ERROR

        set_attribute - True to set, False to clear
        """
        if set_attribute:
            if row_index in self.row_attributes:
                self.row_attributes[row_index].add(attribute)
            else:
                self.row_attributes[row_index] = {attribute}
        else:
            if row_index in self.row_attributes:
                s = self.row_attributes[row_index]
                s.remove(attribute)
                if len(s) == 0:
                    del self.row_attributes[row_index]

    def get_row_attributes(self, row_index):
        """Get the set of attributes on a row

        row_index - index of the row being queried

        returns None if no attributes or a set of attributes set on the row
        """
        return self.row_attributes.get(row_index, None)

    def set_cell_attribute(self, row_index, column_name, attribute, set_attribute=True):
        """Set an attribute on a cell

        row_index - index of row in question

        column_name - name of the cell's column

        attribute - one of the ATTR_ values, for instance ATTR_ERROR

        set_attribute - True to set, False to clear
        """
        key = (row_index, self.column_names.index(column_name))
        if set_attribute:
            if key in self.cell_attributes:
                self.cell_attributes[key].add(attribute)
            else:
                self.cell_attributes[key] = {attribute}
        else:
            if key in self.cell_attributes:
                s = self.cell_attributes[key]
                s.remove(attribute)
                if len(s) == 0:
                    del self.cell_attributes[key]

    def get_cell_attributes(self, row_index, column_name):
        """Get the set of attributes on a row

        row_index - index of the row being queried

        returns None if no attributes or a set of attributes set on the row
        """
        key = (row_index, self.column_names.index(column_name))
        return self.cell_attributes.get(key, None)
