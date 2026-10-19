import os
import tempfile
import logging
import functools
import hashlib
import numpy
import unittest
from urllib.request import URLopener
import skimage.io

import cellprofiler_core.utilities.legacy

LOGGER = logging.getLogger(__name__)

__temp_example_images_folder = None
__temp_test_images_folder = None

cp_logo_url = "https://raw.githubusercontent.com/CellProfiler/CellProfiler/0c64193dd9108934400494fffa41d24f3df1573c/artwork/CP_logo.png"
cp_logo_url_folder, cp_logo_url_filename = cp_logo_url.rsplit("/", 1)
cp_logo_url_shape = (70, 187, 3)


def example_images_directory():
    global __temp_example_images_folder
    if "CP_EXAMPLEIMAGES" in os.environ:
        return os.environ["CP_EXAMPLEIMAGES"]
    fyle = os.path.abspath(__file__)
    d = os.path.split(fyle)[0]  # trunk.CellProfiler.tests.core.modules
    d = os.path.split(d)[0]  # trunk.CellProfiler.tests.core
    d = os.path.split(d)[0]  # trunk.CellProfiler.tests
    d = os.path.split(d)[0]  # trunk.CellProfiler
    d = os.path.split(d)[0]  # trunk
    for imagedir in ["CP-CPEXAMPLEIMAGES", "ExampleImages"]:
        path = os.path.join(d, imagedir)
        if os.path.exists(path):
            return path
    if __temp_example_images_folder is None:
        __temp_example_images_folder = tempfile.mkdtemp(prefix="cp_exampleimages")
        LOGGER.warning(
            "Creating temporary folder %s for example images"
            % __temp_example_images_folder
        )
    return __temp_example_images_folder


def testimages_directory():
    global __temp_test_images_folder
    if "CP_TESTIMAGES" in os.environ:
        return os.environ["CP_TESTIMAGES"]
    fyle = os.path.abspath(__file__)
    d = os.path.split(fyle)[0]  # trunk.CellProfiler.tests.core.modules
    d = os.path.split(d)[0]  # trunk.CellProfiler.tests.core
    d = os.path.split(d)[0]  # trunk.CellProfiler.tests
    d = os.path.split(d)[0]  # trunk.CellProfiler
    d = os.path.split(d)[0]  # trunk
    path = os.path.join(d, "TestImages")
    if os.path.exists(path):
        return path
    if __temp_test_images_folder is None:
        __temp_test_images_folder = tempfile.mkdtemp(prefix="cp_testimages")
        LOGGER.warning(
            "Creating temporary folder %s for test images" % __temp_test_images_folder
        )
    return __temp_test_images_folder


def svn_mirror_url():
    """Return the URL for the SVN mirror

    Use the value of the environment variable, "CP_SVNMIRROR_URL" with
    a default of http://cellprofiler.org/svnmirror.
    """
    return os.environ.get("CP_SVNMIRROR_URL", "https://cellprofiler.org/svnmirror")


def testimages_url():
    return svn_mirror_url() + "/" + "TestImages"


def maybe_download_example_image(folders, file_name, shape=None):
    """Download the given ExampleImages file if not in the directory

    folders - sequence of subfolders starting at ExampleImages
    file_name - name of file to fetch

    Image will be downloaded if not present to CP_EXAMPLEIMAGES directory.

    Returns the local path to the file which is often useful.
    """
    if shape is None:
        shape = (20, 30)
    local_path = os.path.join(
        *tuple([example_images_directory()] + folders + [file_name])
    )
    if not os.path.exists(local_path):
        directory = os.path.join(*tuple([example_images_directory()] + folders))
        if not os.path.isdir(directory):
            os.makedirs(directory)
        random_state = numpy.random.RandomState()
        random_state.seed()
        image = (random_state.uniform(size=shape) * 255).astype(numpy.uint8)
        import skimage.io
        skimage.io.imsave(local_path, image.astype("uint8"))
    return local_path


def make_12_bit_image(folder, filename, shape):
    """Create a 12-bit image of the desired shape

    folder - subfolder of example images directory
    filename - filename for image file
    shape - 2-tuple or 3-tuple of the dimensions of the image. The axis order
            is i, j, c or y, x, c
    """
    r = numpy.random.RandomState()
    r.seed(
        numpy.frombuffer(
            hashlib.sha1("/".join([folder, filename]).encode()).digest(), numpy.uint8
        )
    )
    img = (r.uniform(size=shape) * 4095).astype(numpy.uint16)
    path = os.path.join(example_images_directory(), folder, filename)
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    if len(shape) > 2:
        skimage.io.imsave(path, numpy.transpose(img, (2,0,1)), imagej=True)
    else:
        skimage.io.imsave(path, img)

    #
    # Now go through the file and find the TIF bits per sample IFD (#258) and
    # change it from 16 to 12.
    #
    with open(path, "rb") as fd:
        data = numpy.frombuffer(fd.read(), numpy.uint8).copy()
    offset = numpy.frombuffer(data[4:8].data, numpy.uint32)[0]
    nentries = numpy.frombuffer(data[offset : offset + 2], numpy.uint16)[0]
    ifds = []
    # Get the IFDs we don't modify
    for idx in range(nentries):
        ifd = data[offset + 2 + idx * 12 : offset + 14 + idx * 12]
        code = ifd[0] + 256 * ifd[1]
        if code not in (258, 281):
            ifds.append(ifd)
    ifds += [
        # 12 bits/sample
        numpy.array([2, 1, 3, 0, 1, 0, 0, 0, 12, 0, 0, 0], numpy.uint8),
        # max value = 4095
        numpy.array([25, 1, 3, 0, 1, 0, 0, 0, 255, 15, 0, 0], numpy.uint8),
    ]
    ifds = sorted(
        ifds,
        key=functools.cmp_to_key(
            lambda a, b: cellprofiler_core.utilities.legacy.cmp(a.tolist(), b.tolist())
        ),
    )
    old_end = offset + 2 + nentries * 12
    new_end = offset + 2 + len(ifds) * 12
    diff = new_end - old_end
    #
    # Fix up the IFD offsets if greater than "offset"
    #
    for ifd in ifds:
        count = numpy.frombuffer(ifd[4:8].data, numpy.uint32)[0]
        if count > 4:
            ifd_off = (
                numpy.array([numpy.frombuffer(ifd[8:12].data, numpy.uint32)[0]]) + diff
            )
            if ifd_off > offset:
                ifd[8:12] = numpy.frombuffer(ifd_off.data, numpy.uint8)
    new_data = numpy.zeros(len(data) + diff, numpy.uint8)
    new_data[:offset] = data[:offset]
    new_data[offset] = len(ifds) % 256
    new_data[offset + 1] = int(len(ifds) / 256)
    for idx, ifd in enumerate(ifds):
        new_data[offset + 2 + idx * 12 : offset + 14 + idx * 12] = ifd
    new_data[new_end:] = data[old_end:]

    with open(path, "wb") as fd:
        fd.write(new_data.data)
    return path


def maybe_download_example_images(folders, file_names):
    """Download multiple files to the example images directory

    folders - sequence of subfolders of ExampleImages
    file_names - sequence of file names to be fetched from the single directory
                described by the list of folders

    Returns the local directory containing the images.
    """
    for file_name in file_names:
        maybe_download_example_image(folders, file_name)
    return os.path.join(example_images_directory(), *folders)


def maybe_download_fly():
    """Download the fly example directory"""
    return maybe_download_example_images(
        ["ExampleFlyImages"],
        [
            "01_POS002_D.TIF",
            "01_POS002_F.TIF",
            "01_POS002_R.TIF",
            "01_POS076_D.TIF",
            "01_POS076_F.TIF",
            "01_POS076_R.TIF",
            "01_POS218_D.TIF",
            "01_POS218_F.TIF",
            "01_POS218_R.TIF",
        ],
    )


def maybe_download_tesst_image(file_name):
    """Download the given TestImages file if not in the directory

    file_name - name of file to fetch

    Image will be downloaded if not present to CP_EXAMPLEIMAGES directory.
    """
    local_path = os.path.join(testimages_directory(), file_name)
    if not os.path.exists(local_path):
        url = testimages_url() + "/" + file_name
        try:
            URLopener().retrieve(url, local_path)
        except IOError as e:
            # This raises the "expected failure" exception.
            def bad_url(e=e):
                raise e

            unittest.expectedFailure(bad_url)()
    return local_path


def read_example_image(folder, file_name, **kwargs):
    """Read an example image from one of the example image directories

    folder - folder containing images, e.g., "ExampleFlyImages"

    file_name - the name of the file within the folder

    **kwargs - any keyword arguments are passed onto load_image
    """
    import imageio

    path = os.path.join(example_images_directory(), folder, file_name)
    maybe_download_example_image([folder], file_name)
    return imageio.imread(path, **kwargs)


raw_8_1 = "AAQJDRIWGx8kKC0xNjo/Q0hMUVZaX2NobHF1en6Dh4wEBgoOEhcbICQpLTI2Oz9ESE1RVlpfY2hscXV6foOHjAkKDBAUGBwgJSkuMjc7QERJTVJWW19kaG1xdnp/g4iMDQ4QExYaHiImKi8zODxARUlOUldbYGRpbXJ2e3+EiI0SEhQWGR0gJCgsMDU5PUFGSk9TV1xgZWlucnd7gISJjRYXGBodICMmKi4yNjo/Q0dLUFRYXWFmam9zd3yAhYmOGxscHiAjJiktMDQ4PEBESU1RVVpeYmdrcHR4fYGGio8fICAiJCYpLDAzNzs+QkZKT1NXW19kaGxxdXl+goeLjyQkJSYoKi0wMzY6PUFFSUxRVVldYWVqbnJ2e3+DiIyRKCkpKiwuMDM2OTxAQ0dLT1NXW19jZ2tvdHh8gIWJjZItLS4vMDI0Nzo8QENGSk1RVVldYWVpbXF1eX6ChoqPkzEyMjM1Njg7PUBDRklNUFRXW19jZ2tvc3d7f4SIjJCUNjY3ODk6PD5BQ0ZJTFBTV1peYWVpbXF1eX2BhYmOkpY6Ozs8PT9AQkVHSk1QU1ZZXWBkaGxvc3d7f4OHi4+UmD8/QEBBQ0RGSUtNUFNWWVxgY2dqbnJ2eX2BhYmNkZWaQ0RERUZHSUpMT1FUV1lcYGNmam1xdHh8gISHi4+Tl5tISElJSktNT1FTVVdaXWBjZmltcHR3e36ChoqOkpaZnUxNTU5PUFFTVVdZW15gY2ZpbHBzdnp9gYWIjJCUmJygUVFSUlNUVVdZW11fYWRnam1wc3Z5fYCEh4uPkpaanqJWVlZXV1haW11fYWNlaGptcHN2eXyAg4eKjpGVmZ2gpFpaW1tcXV5fYWNlZ2lsbnF0dnl8gIOGio2RlJibn6OnX19fYGBhYmRlZ2lrbW9ydHd6fYCDhomNkJOXmp6ipaljY2RkZWZnaGprbW9xc3Z4e32Ag4aJjJCTlpqdoaSorGhoaGlpamtsbm9xc3V3eXx+gYSHio2Qk5aZnaCkp6uubGxtbW5vcHFydHV3eXt9gIKFh4qNkJOWmZygo6eqrrFxcXFycnN0dXZ4eXt9f4GEhoiLjpGTlpmcoKOmqq2wtHV1dnZ3d3h5e3x+f4GDhYeKjI+RlJeanaCjpqmtsLO3enp6e3t8fX5/gIKEhYeJi46QkpWYmp2go6aprLCztrp+fn9/gICBgoOFhoiJi42PkpSWmZueoaSnqq2ws7a5vYODg4SEhYaHiImKjI6PkZOWmJqdn6Kkp6qtsLO2ubzAh4eIiImJiouMjY+QkpSVl5mcnqCjpairrrCztrm8wMOMjIyNjY6Pj5GSk5SWmJqbnaCipKeprK6xtLe6vcDDxpCRkZGSkpOUlZaXmZqcnqCipKaoqq2vsrW3ur3Aw8bJlZWVlpaXl5iZmpydn6CipKaoqqyusbO2uLu+wcTGyc2Zmpqam5ucnZ6foKGjpKaoqqyusLK0t7m8v8HEx8rN0J6enp+foKChoqOkpqepqqyusLK0tri7vcDCxcjLzdDTo6Ojo6SkpaanqKmqq62usLK0tri6vL/Bw8bJy87R1Nenp6eoqKmpqqusra6wsbO0tri6vL7AwsXHyszP0tTX2qysrKytra6vr7Cxs7S1t7m6vL7AwsTGycvO0NPV2NvesLCwsbGysrO0tba3uLq7vb7AwsTGyMrNz9HU1tnc3uG1tbW1tra3t7i5uru9vr/Bw8TGyMrMztDT1dja3d/i5bm5ubq6u7u8vb6/wMHCxMXHycrMztDS1NfZ297g4+bovr6+vr+/wMDBwsPExcfIycvNztDS1NbY293f4uTn6ezCwsLDw8TExcbGx8nKy8zOz9HT1NbY2tzf4ePl6Ort8MfHx8fIyMnJysvMzc7P0dLU1dfZ2tze4OLl5+ns7vHzy8vMzMzNzc7Pz9DR0tTV1tjZ293f4OLk5unr7fDy9PfQ0NDQ0dHS0tPU1dbX2Nnb3N7f4ePl5ujq7e/x8/b4+9TU1dXV1tbX19jZ2tvc3t/g4uPl5+nr7O/x8/X3+vz/"
raw_8_1_shape = (48, 32)
raw_8_2 = "//v28u3p5ODb19LOycXAvLezrqmloJyXk46KhYF8eHP7+fXx7ejk39vW0s3JxMC7t7KuqaWgnJeTjoqFgXx4c/b18+/r5+Pf2tbRzcjEv7u2sq2ppKCbl5KOiYWAfHdz8vHv7Onl4d3Z1dDMx8O/uraxraikn5uWko2JhIB7d3Lt7evp5uLf29fTz8rGwr65tbCsqKOfmpaRjYiEf3t2cuno5+Xi39zZ1dHNycXAvLi0r6unop6ZlZCMiIN/enZx5OTj4d/c2dbSz8vHw7+7trKuqqWhnZiUj4uHgn55dXDg39/d29nW08/MyMTBvbm1sKyopKCbl5OOioaBfXh0cNvb2tnX1dLPzMnFwr66trOuqqainpqVkY2JhIB8d3Nu19bW1dPRz8zJxsO/vLi0sKyopKCcmJSQi4eDf3p2cm3S0tHQz83LyMXDv7y5tbKuqqainpqWko6KhoF9eXVwbM7NzczKycfEwr+8ubayr6uopKCcmJSQjIiEgHt3c29rycnIx8bFw8G+vLm2s6+sqKWhnpqWko6KhoJ+enZxbWnFxMTDwsC/vbq4tbKvrKmmop+bl5OQjIiEgHx4dHBrZ8DAv7++vLu5trSyr6yppqOfnJiVkY2JhoJ+enZybmplvLu7urm4trWzsK6rqKajn5yZlZKOi4eDf3t4dHBsaGS3t7a2tbSysK6sqqilop+cmZaSj4uIhIF9eXVxbWlmYrOysrGwr66sqqimpKGfnJmWk4+MiYWCfnp3c29rZ2Nfrq6trayrqqimpKKgnpuYlZKPjImGgn97eHRwbWllYV2pqamoqKelpKKgnpyal5WSj4yJhoN/fHh1cW5qZmJfW6WlpKSjoqGgnpyamJaTkY6LiYaDf3x5dXJua2dkYFxYoKCgn5+enZuamJaUkpCNi4iFgn98eXZyb2xoZWFdWlacnJubmpmYl5WUkpCOjImHhIJ/fHl2c29saWViXltXU5eXl5aWlZSTkZCOjIqIhoOBfnt4dXJvbGlmYl9bWFRRk5OSkpGQj46Ni4qIhoSCf316eHVyb2xpZmNfXFhVUU6Ojo6NjYyLiomHhoSCgH57eXd0cW5saWZjX1xZVVJPS4qKiYmIiIeGhIOBgH58enh1c3Bua2hlYl9cWVZST0xIhYWFhISDgoGAf317enh2dHFvbWpnZWJfXFlWU09MSUWBgYCAf39+fXx6eXd2dHJwbWtpZmRhXltYVVJPTElGQnx8fHt7enl4d3Z1c3FwbmxpZ2ViYF1bWFVST0xJRkM/eHh3d3Z2dXRzcnBvbWtqaGZjYV9cWldUUU9MSUZDPzxzc3NycnFwcG5tbGtpZ2VkYl9dW1hWU1FOS0hFQj88OW9ubm5tbWxramloZmVjYV9dW1lXVVJQTUpIRUI/PDk2ampqaWloaGdmZWNiYF9dW1lXVVNRTkxJR0RBPjs5NjJmZWVlZGRjYmFgX15cW1lXVVNRT01LSEZDQD47ODUyL2FhYWBgX19eXVxbWVhWVVNRT01LSUdEQj89Ojc0Mi8sXFxcXFtbWllYV1ZVVFJRT01LSUdFQ0A+PDk2NDEuKyhYWFhXV1ZWVVRTUlFPTkxLSUdFQ0E/PTo4NTMwLSsoJVNTU1NSUlFQUE9OTEtKSEZFQ0E/PTs5NjQxLywqJyQhT09PTk5NTUxLSklIR0VEQkE/PTs5NzUyMC4rKSYjIR5KSkpKSUlISEdGRURCQUA+PDs5NzUzMS8sKiclIiAdGkZGRkVFRERDQkFAPz49Ozo4NjUzMS8tKygmJCEfHBkXQUFBQUBAPz8+PTw7Ojg3NjQyMS8tKyknJCIgHRsYFhM9PT08PDs7Ojk5ODY1NDMxMC4sKyknJSMgHhwaFxUSDzg4ODg3NzY2NTQzMjEwLi0rKigmJSMhHx0aGBYTEQ4MNDQzMzMyMjEwMC8uLSsqKScmJCIgHx0bGRYUEg8NCwgvLy8vLi4tLSwrKikoJyYkIyEgHhwaGRcVEhAODAkHBCsrKioqKSkoKCcmJSQjISAfHRwaGBYUExAODAoIBQMA"
raw_8_2_shape = (48, 32)
tif_8_1 = "SUkqAAgAAAAIAAABAwABAAAAIAAAAAEBAwABAAAAMAAAAAIBAwABAAAACAAAAAMBAwABAAAAAQAAAAYBAwABAAAAAQAAABEBBAABAAAAbgAAABYBAwABAAAAMAAAABcBAwABAAAAAAYAAAAAAAAABAkNEhYbHyQoLTE2Oj9DSExRVlpfY2hscXV6foOHjAQGCg4SFxsgJCktMjY7P0RITVFWWl9jaGxxdXp+g4eMCQoMEBQYHCAlKS4yNztARElNUlZbX2RobXF2en+DiIwNDhATFhoeIiYqLzM4PEBFSU5SV1tgZGltcnZ7f4SIjRISFBYZHSAkKCwwNTk9QUZKT1NXXGBlaW5yd3uAhImNFhcYGh0gIyYqLjI2Oj9DR0tQVFhdYWZqb3N3fICFiY4bGxweICMmKS0wNDg8QERJTVFVWl5iZ2twdHh9gYaKjx8gICIkJiksMDM3Oz5CRkpPU1dbX2RobHF1eX6Ch4uPJCQlJigqLTAzNjo9QUVJTFFVWV1hZWpucnZ7f4OIjJEoKSkqLC4wMzY5PEBDR0tPU1dbX2Nna290eHyAhYmNki0tLi8wMjQ3OjxAQ0ZKTVFVWV1hZWltcXV5foKGio+TMTIyMzU2ODs9QENGSU1QVFdbX2Nna29zd3t/hIiMkJQ2Njc4OTo8PkFDRklMUFNXWl5hZWltcXV5fYGFiY6Sljo7Ozw9P0BCRUdKTVBTVlldYGRobG9zd3t/g4eLj5SYPz9AQEFDREZJS01QU1ZZXGBjZ2pucnZ5fYGFiY2RlZpDRERFRkdJSkxPUVRXWVxgY2ZqbXF0eHyAhIeLj5OXm0hISUlKS01PUVNVV1pdYGNmaW1wdHd7foKGio6SlpmdTE1NTk9QUVNVV1lbXmBjZmlscHN2en2BhYiMkJSYnKBRUVJSU1RVV1lbXV9hZGdqbXBzdnl9gISHi4+SlpqeolZWVldXWFpbXV9hY2Voam1wc3Z5fICDh4qOkZWZnaCkWlpbW1xdXl9hY2VnaWxucXR2eXyAg4aKjZGUmJufo6dfX19gYGFiZGVnaWttb3J0d3p9gIOGiY2Qk5eanqKlqWNjZGRlZmdoamttb3Fzdnh7fYCDhomMkJOWmp2hpKisaGhoaWlqa2xub3FzdXd5fH6BhIeKjZCTlpmdoKSnq65sbG1tbm9wcXJ0dXd5e32AgoWHio2Qk5aZnKCjp6qusXFxcXJyc3R1dnh5e31/gYSGiIuOkZOWmZygo6aqrbC0dXV2dnd3eHl7fH5/gYOFh4qMj5GUl5qdoKOmqa2ws7d6enp7e3x9fn+AgoSFh4mLjpCSlZianaCjpqmssLO2un5+f3+AgIGCg4WGiImLjY+SlJaZm56hpKeqrbCztrm9g4ODhISFhoeIiYqMjo+Rk5aYmp2foqSnqq2ws7a5vMCHh4iIiYmKi4yNj5CSlJWXmZyeoKOlqKuusLO2ubzAw4yMjI2Njo+PkZKTlJaYmpudoKKkp6msrrG0t7q9wMPGkJGRkZKSk5SVlpeZmpyeoKKkpqiqra+ytbe6vcDDxsmVlZWWlpeXmJmanJ2foKKkpqiqrK6xs7a4u77BxMbJzZmampqbm5ydnp+goaOkpqiqrK6wsrS3uby/wcTHys3Qnp6en5+goKGio6Smp6mqrK6wsrS2uLu9wMLFyMvN0NOjo6OjpKSlpqeoqaqrra6wsrS2uLq8v8HDxsnLztHU16enp6ioqamqq6ytrrCxs7S2uLq8vsDCxcfKzM/S1NfarKysrK2trq+vsLGztLW3ubq8vsDCxMbJy87Q09XY296wsLCxsbKys7S1tre4uru9vsDCxMbIys3P0dTW2dze4bW1tbW2tre3uLm6u72+v8HDxMbIyszO0NPV2Nrd3+Llubm5urq7u7y9vr/AwcLExcfJyszO0NLU19nb3uDj5ui+vr6+v7/AwMHCw8TFx8jJy83O0NLU1tjb3d/i5Ofp7MLCwsPDxMTFxsbHycrLzM7P0dPU1tja3N/h4+Xo6u3wx8fHx8jIycnKy8zNzs/R0tTV19na3N7g4uXn6ezu8fPLy8zMzM3Nzs/P0NHS1NXW2Nnb3d/g4uTm6evt8PL099DQ0NDR0dLS09TV1tfY2dvc3t/h4+Xm6Ort7/Hz9vj71NTV1dXW1tfX2Nna29ze3+Di4+Xn6evs7/Hz9ff6/P8="
jpg_8_1 = "/9j/4AAQSkZJRgABAQEAYABgAAD/4QB2RXhpZgAASUkqAAgAAAAIAAABAwABAAAAIADM/wEBAwABAAAAMADP/wIBAwABAAAACADS/wMBAwABAAAAAQDV/wYBAwABAAAAAQDY/xEBBAABAAAAbgAAABYBAwABAAAAMADe/xcBAwABAAAAAAbh/wAAAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAAwACADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD+MLwR4In0+eNVjZVVhgYPHPQdun+cV96fDHSJozbhkI5XqPp/n09TyaNI+GJjmRhbnqD93+uP8jJ5619MeBfArQtD+5IwVHCnnJHtjp0FAH0B8LrGRfs4Kkfc9fYn9cc//Xz+jvwttGH2fg8bP6f/AFv89fkj4ceFnQwZjPG3tg/56981+gXwz0BkNv8AIQfl7fQjI7f/AKs0AfkvY/C7bIpFv0P9z09yM/Xj/E+zeFvhwUeM+Rjkfw9x/k9+tfW9p8LcMP8AR8c/3P8A639Py7+l6B8MyjJi35BGfl/I4I7f/XxQB5f4E8CMjQ/uTjKjhT69Tx/j1r7d+Hng9kMH7o4+X+Hj/PB9AKPB/wAOyjRfuOMr/D27en8+AOlfXngXwKUMIEJ6r/DjGOvPf+tAHhdv8LfmH+j9x/D/AJ/mPw793ovwx2smLfuP4Mf0J/HPXHpX3LB8Lef+Pb/xz8Pf+v8AQ9hpXwvwy/6P37IP8OvueB1xQB8u+GPhvhoj9n9P4Py7euK+n/Bnw+KmL9x3U/d/n0/U9c8V7J4e+GZBQi39M/J+vTv3/wA4+ivCfw5KtF/o/wDdONvr+HY/zNAH/9k="
png_8_1 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAwCAAAAABVxyezAAADHElEQVR4nGJkYGFFhygcAAAAAP//YmJhYmRkQIUoAAAAAP//YmFlZGLCVIOAAAAAAP//YmFhZGJmYmJiYmRkZMSmDgAAAP//YmFlYGJihiiBKkJVAQAAAP//YmFhZGRihqpghBmEpA4AAAD//2JhZWBkYmJGUYKsihEAAAD//2JhYWRggqhgZmZiYoI4GaGEAQAAAP//YmFlYIQawQyziokJYRATAAAA//9iYWFkYIQYwcTMxMQEUwFXAgAAAP//YmFlYIQYwcQCNwBZCQAAAP//gpjAyMQMdQayEiYmJkZGAAAAAP//gpjAyIRQwYxiCBMAAAD//2JhYWRgRDKCCcmpTEyMTEwAAAAA//9iYWWA2YFkBFwJIxMAAAD//4KYADOCiZkFrgSihhEAAAD//4KYwAhRz4wATLDgBQAAAP//gpoAUQFzKMQ0iBYAAAAA//+CmoDFCKgSAAAAAP//YmFlgLoS2QikkAcAAAD//2JhYUCzg4kZ6lKIWwEAAAD//4KagMUIqEMAAAAA//9CmMAIVYBmCwAAAP//gpmA1RXMzMxMAAAAAP//gpuA3QhmZgAAAAD//4KbAFfABPUAC8QOAAAAAP//gpgAcSVWIwAAAAD//0KYwAhJAZCoQigAAAAA//+CmwAzghnNoQAAAAD//0IyAWIEM5otAAAAAP//QpjAyISsAmoNCzMAAAD//4KZwMAECW5GdCMAAAAA//9iYWFkQA0JpDhhZmZmBgAAAP//gpqA5EpkvzIxswAAAAD//0I2AasRAAAAAP//QjIBixFMzMwAAAAA//9iYWVAmICsAGYEAAAA//9iYUGYgGkEEzMzAAAA//+Cm8DAyMgEC26YAiZmZmYAAAAA//9CMQHJCGaYQwEAAAD//4KZAHUlphEAAAAA//9CNQEW3EhGAAAAAP//QjEB2QhYiAMAAAD//0KYwMDIhGwEMzS4AAAAAP//QjUB6kpGJCMAAAAA//9iYWFghPuTkQkaVvCUw8QMAAAA//9CMwHmUUa4HQAAAAD//4KbgHAlIxMTExMjM9SvAAAAAP//AwAHWgwSjKsCjAAAAABJRU5ErkJggg=="
gif_8_1 = "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"
tif_8_2 = "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"
jpg_8_2 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAwACABAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/APZ7m5DjrWBeyA5rnL1hzXN3zda9HkvcjrWZdXWc81hXk+c81z97L1rrWveOtUp7zIPNZN1dZzzWHd3HXmttr73qtLe+9Ztxd9eayLq6681fN971XkvfeqM9515rLuLvrzUxvveoJL33qnLe+9Z09515r//Z"
png_8_2 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAwCAAAAABVxyezAAADMklEQVR4nGL8/+c3OkThAAAAAP//Yvrz7/9/BlSIAgAAAAD//2L5/f/fP0w1CAgAAAD//2L58//f33///v37////f2zqAAAAAP//YvnN8O/fX4gSqCJUFQAAAAD//2L58///v79QFf9hBiGpAwAAAP//YvnN8P/fv78oSpBV/QcAAAD//2L585/hH0TF37///v2DOBmhhAEAAAD//2L5zfAfasRfmFX//iEM+gcAAAD//2L585/hP8SIf3///fsHUwFXAgAAAP//YvnN8B9ixL8/cAOQlQAAAAD//4KY8P/fX6gzkJX8+/fv/38AAAAA//+CmPD/H0LFXxRD/gEAAAD//2L585/hP5IR/5Cc+u/f/3//AAAAAP//YvnNALMDyQi4kv//AAAAAP//gpgAM+Lf3z9wJRA1/wEAAAD//4KY8B+i/i8C/IMFLwAAAP//gpoAUQFzKMQ0iBYAAAAA//+CmoDFCKgSAAAAAP//YvnNAHUlshFIIQ8AAAD//2L5w4Bmx7+/UJdC3AoAAAD//4KagMUIqEMAAAAA//9CmPAfqgDNFgAAAAD//4KZgNUVf//+/QcAAAD//4KbgN2Iv38BAAAA//+CmwBX8A/qgT8QOwAAAAD//4KYAHElViMAAAAA//9CmPAfkgIgUYVQAAAAAP//gpsAM+IvmkMBAAAA//9CMgFixF80WwAAAAD//0KY8P8fsgqoNX/+AgAAAP//gpnA8A8S3P/RjQAAAAD//2L5858BNSSQ4uTv379/AQAAAP//gpqA5Epkv/77+wcAAAD//0I2AasRAAAAAP//QjIBixH//v4FAAAA//9i+c2AMAFZAcwIAAAAAP//YvmDMAHTiH9//wIAAAD//4KbwPD//z9YcMMU/Pv79y8AAAD//0IxAcmIvzCHAgAAAP//gpkAdSWmEQAAAAD//0I1ARbcSEYAAAAA//9CMQHZCFiIAwAAAP//QpjA8P8fshF/ocEFAAAA//9CNQHqyv9IRgAAAAD//2L5w/Af7s///6BhBU85//4CAAAA//9CMwHm0f9wOwAAAAD//4KbgHDl/3///v37/xfqVwAAAAD//wMAip9DsGSZAhgAAAAASUVORK5CYII="
gif_8_2 = "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"

github_url = "https://github.com/CellProfiler/CellProfiler/raw/master"
