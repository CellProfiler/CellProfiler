import importlib
import inspect
import sys
import warnings

MESSAGE = "This package is deprecated, please use centrosome"

warnings.warn(MESSAGE, DeprecationWarning)


def bind(module):
    k = module.rsplit(".")[-1]
    module = sys.modules[module]

    warnings.warn(MESSAGE + ".%s" % k, DeprecationWarning, stacklevel=2)

    package = importlib.import_module("centrosome." + k)

    for k, v in inspect.getmembers(package):
        if inspect.getmodule(v) in (None, package):
            setattr(module, k, v)
