import os
import urllib2
import gzip
import StringIO


def retrieve_prokaryote(version='1.0.0'):
    url = 'https://github.com/CellProfiler/prokaryote/releases/download/{0}/prokaryote-{0}.tar.gz'.format(version)

    path = './imagej/jars'

    filename = '{0}/prokaryote-{1}.jar'.format(path, version)

    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

    if not os.path.isfile(filename):
        response = urllib2.urlopen(url)

        with open(filename, 'w') as outfile:
            outfile.write(gzip.GzipFile(fileobj=(StringIO.StringIO(response.read()))).read())
