class GroupModule(cpm.CPModule):
    module_name = "Group"
    variable_revision_number = 1

    def setup(self, groupings,
              prepare_run_callback=None,
              prepare_group_callback=None,
              run_callback=None,
              post_group_callback=None,
              post_run_callback=None,
              get_measurement_columns_callback=None,
              display_callback=None,
              display_post_group_callback=None,
              display_post_run_callback=None):
        self.prepare_run_callback = prepare_run_callback
        self.prepare_group_callback = prepare_group_callback
        self.run_callback = run_callback
        self.post_group_callback = post_group_callback
        self.post_run_callback = post_run_callback
        self.groupings = groupings
        self.get_measurement_columns_callback = get_measurement_columns_callback
        self.display_callback = None
        self.display_post_group_callback = None
        self.display_post_run_callback = None

    def settings(self):
        return []

    def get_groupings(self, workspace):
        return self.groupings

    def prepare_run(self, *args):
        if self.prepare_run_callback is not None:
            return self.prepare_run_callback(*args)
        return True

    def prepare_group(self, *args):
        if self.prepare_group_callback is not None:
            return self.prepare_group_callback(*args)
        return True

    def run(self, *args):
        if self.run_callback is not None:
            return self.run_callback(*args)

    def post_run(self, *args):
        if self.post_run_callback is not None:
            return self.post_run_callback(*args)

    def post_group(self, *args):
        if self.post_group_callback is not None:
            return self.post_group_callback(*args)

    def get_measurement_columns(self, *args):
        if self.get_measurement_columns_callback is not None:
            return self.get_measurement_columns_callback(*args)
        return []

    def display(self, workspace, figure):
        if self.display_callback is not None:
            return self.display_callback(workspace, figure)
        return super(GroupModule, self).display(workspace, figure)

    def display_post_group(self, workspace, figure):
        if self.display_post_group_callback is not None:
            return self.display_post_group_callback(workspace, figure)
        return super(GroupModule, self).display_post_group(workspace, figure)

    def display_post_run(self, workspace, figure):
        if self.display_post_run is not None:
            return self.display_post_run_callback(workspace, figure)
        return super(GroupModule, self).display_post_run(workspace, figure)

