import os
import urllib


def retrieve_prokaryote(version='1.0.0'):
    url = 'https://github.com/CellProfiler/prokaryote/releases/download/{0}/prokaryote-{0}.jar'.format(version)

    path = './imagej/jars'

    filename = '{0}/prokaryote-{1}.jar'.format(path, version)

    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

    if not os.path.isfile(filename):
        urllib.urlretrieve(url, filename)
