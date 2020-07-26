from ._setting import Setting


class BinaryMatrix(Setting):
    """A setting that allows editing of a 2D matrix of binary values
    """

    def __init__(
        self, text, default_value=True, default_width=5, default_height=5, **kwargs
    ):
        initial_value_text = self.to_value(
            [[default_value] * default_width] * default_height
        )
        Setting.__init__(self, text, initial_value_text, **kwargs)

    @staticmethod
    def to_value(matrix):
        """Convert a matrix to a pickled form

        format is <row-count>,<column-count>,<0 or 1>*row-count*column-count

        e.g., [[True, False, True], [True, True, True]] -> "2,3,101111"
        """
        h = len(matrix)
        w = 0 if h == 0 else len(matrix[0])
        return ",".join(
            (
                str(h),
                str(w),
                "".join(["".join(["1" if v else "0" for v in row]) for row in matrix]),
            )
        )

    def get_matrix(self):
        """Return the setting's matrix"""
        hs, ws, datas = self.value_text.split(",")
        h, w = int(hs), int(ws)
        return [[datas[i * w + j] == "1" for j in range(w)] for i in range(h)]

    def get_size(self):
        """Return the size of the matrix

        returns a tuple of height, width
        """
        hs, ws, datas = self.value_text.split(",")
        return int(hs), int(ws)
