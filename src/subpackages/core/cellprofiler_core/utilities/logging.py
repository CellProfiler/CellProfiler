import logging

def set_log_level(log_level, subprocess=False):
    """Set the logging package's log level based on command-line options"""
    try:
        if log_level.isdigit():
            logging.root.setLevel(int(log_level))
        else: # str, e.g. str(logging.INFO) -> '20' OR 'INFO'
            logging.root.setLevel(log_level)

        if len(logging.root.handlers) == 0:
            stream_handler = logging.StreamHandler()
            if subprocess:
                fmt = logging.Formatter("%(process)d|%(levelno)s|%(name)s::%(funcName)s: %(message)s")
            else:
                fmt = logging.Formatter("[CP - %(levelname)s] %(name)s::%(funcName)s: %(message)s")
            stream_handler.setFormatter(fmt)
            logging.root.addHandler(stream_handler)

        # silence matplotlib debug messages, they're super annoying
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
    except ValueError as e:
        logging.config.fileConfig(log_level)
