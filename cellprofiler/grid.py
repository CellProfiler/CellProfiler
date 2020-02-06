class Grid:
    def __init__(self):
        self.x_location_of_lowest_x_spot = None
        self.y_location_of_lowest_y_spot = None
        self.x_spacing = None
        self.y_spacing = None
        self.rows = None
        self.columns = None
        self.vert_lines_x = None
        self.vert_lines_y = None
        self.horiz_lines_x = None
        self.horiz_lines_y = None
        self.spot_table = None
        self.total_height = None
        self.total_width = None
        self.y_locations = None
        self.x_locations = None
        self.left_to_right = None
        self.top_to_bottom = None
        self.image_width = None
        self.image_height = None

    def serialize(self):
        return dict(
            (k, v) for k, v in list(self.__dict__.items()) if not k.startswith("_")
        )

    def deserialize(self, serialized_info):
        self.__dict__.update(serialized_info)
