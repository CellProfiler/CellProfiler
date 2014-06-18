#!/usr/bin/env python

import cellprofiler.utilities.version

f = open("cellprofiler/frozen_version.py", "w")
f.write("# MACHINE_GENERATED\nversion_string = '%s'" % cellprofiler.utilities.version.version_string)
f.close()

