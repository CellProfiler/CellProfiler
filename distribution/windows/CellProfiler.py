import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    import cellprofiler.__main__

    sys.exit(cellprofiler.__main__.main())
