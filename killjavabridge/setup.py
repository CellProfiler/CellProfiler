"""killjavabridge setup.py - easy_install setup to kill the javabridge

This module creates a Nose plugin that can kill the Java VM
started by cellprofiler.utilities.jutil. It should be installed using easy_install

"""

__version__="$Revision$"

import os

def configuration():
    config = dict( 
        name = "killjavabridge",
        version = "0.1",
        description = "Nose plugin to kill the Java bridge after test completion",
        maintainer = "Lee Kamentsky",
        maintainer_email = "leek@broadinstitute.org",
        packages = ["killjavabridge"],
        entry_points = {
            'nose.plugins.0.10' : 
            'kill-vm = killjavabridge:KillVMPlugin'
        })
    return config

def main():
    from setuptools import setup

    setup(**configuration())
    
if __name__ == '__main__':
    main()


