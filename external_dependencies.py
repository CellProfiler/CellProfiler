import os
import urllib

VERSION = "1.0.4"

directory = "imagej/jars"

if not os.path.exists(directory):
    os.makedirs(directory)

prokaryote = '{}/prokaryote-{}.jar'.format(os.path.abspath("imagej/jars"), VERSION)

if not os.path.isfile(prokaryote):
    urllib.urlretrieve('https://github.com/CellProfiler/prokaryote/' + 'releases/download/{tag}/prokaryote-{tag}.jar'.format(tag=VERSION), prokaryote)

dependencies = os.path.abspath('imagej/jars/cellprofiler-java-dependencies-classpath.txt')

if not os.path.isfile(dependencies):
    file = open(dependencies, 'w')

    file.write(prokaryote)

    file.close()
