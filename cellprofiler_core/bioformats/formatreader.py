K_OMERO_SERVER = None
K_OMERO_PORT = None
K_OMERO_USER = None
K_OMERO_SESSION_ID = None
K_OMERO_CONFIG_FILE = None

def get_image_reader():
    raise RuntimeError("unimplemented")

def release_image_reader():
    raise RuntimeError("unimplemented")

def clear_image_reader_cache():
    raise RuntimeError("unimplemented")

def set_omero_login_hook(login):
    raise RuntimeError("unimplemented")

def get_omero_credentials():
    raise RuntimeError("unimplemented")

def use_omero_credentials(credentials):
    raise RuntimeError("unimplemented")
