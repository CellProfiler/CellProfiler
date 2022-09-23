import multiprocessing
import sys
if hasattr(sys, "frozen"):
    print("Frozen app detected", __name__)
    # allows multiprocessing on Windows after freeze
    multiprocessing.freeze_support()

print("Testing imports")
import skimage.filters.rank
print("Skimage OK")
import cellprofiler_core.constants.object
print("Core OK")
import cellprofiler.gui.constants.preferences_view
print("GUI OK")
import cellprofiler.__main__
print("Main OK")

if __name__ == "__main__":
    cellprofiler.__main__.main()
