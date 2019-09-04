Make sure build machine has pywin32 and InnoSetup installed

To Run:
* clone CellProfiler at correct version into the `windows` directory
* run `pyinstaller CellProfiler.spec` 

* Update CellProfiler.iss to correct version number (and correct Java, if necessary)
* Run InnoSetup on .iss file.
