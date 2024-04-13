import multiprocessing

if __name__ == "__main__":
    # important to enable freeze_support when creating an executable (i.e. via pyinstaller):
    # https://docs.python.org/3.9/library/multiprocessing.html#multiprocessing.freeze_support
    #
    # allows multiprocessing on Windows after freeze:
    # https://github.com/pyinstaller/pyinstaller/wiki/Recipe-Multiprocessing
    #
    # despite what the python docs say, it's also important on macOS,
    # putting `import cellprofiler.__main__`` above freeze_support causes a continuous re-spawning of
    # cellprofiler after building with pyinstaller on MacOS
    # most likely because it seems pyinstaller monkey patches `freeze_support`:
    # https://github.com/pyinstaller/pyinstaller/blob/develop/PyInstaller/hooks/rthooks/pyi_rth_multiprocessing.py
    multiprocessing.freeze_support()

    # __main__ has many top level imports, eventually running into imports of joblib
    # we always conditionally import joblib, but we have some top level imports of sk_learn
    # which itself has top level joblib imports
    # these must come after freeze_support() is enabled:
    # https://github.com/pyinstaller/pyinstaller/issues/4110
    import cellprofiler.__main__

    cellprofiler.__main__.main()
