__the_workspace_viewer = None


def show_workspace_viewer(parent, workspace):
    global __the_workspace_viewer
    if __the_workspace_viewer is None:
        __the_workspace_viewer = Frame(parent, workspace)
    else:
        __the_workspace_viewer.set_workspace(workspace)
        __the_workspace_viewer.frame.Show()


def update_workspace_viewer(workspace):
    if __the_workspace_viewer is not None:
        __the_workspace_viewer.set_workspace(workspace)
