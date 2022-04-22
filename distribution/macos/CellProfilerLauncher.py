import multiprocessing
import sys
if hasattr(sys, "frozen"):
    print("Frozen app detected", __name__)
    # allows multiprocessing on Windows after freeze
    multiprocessing.freeze_support()

import cellprofiler.__main__

if __name__ == "__main__":
    cellprofiler.__main__.main()
