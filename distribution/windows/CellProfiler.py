import multiprocessing

import cellprofiler.__main__

if __name__ == "__main__":
    # allows multiprocessing on Windows after freeze
    multiprocessing.freeze_support()
    cellprofiler.__main__.main()
