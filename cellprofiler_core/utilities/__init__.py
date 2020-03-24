import os
import re

import boto3
import pkg_resources


def image_resource(filename):
    relpath = os.path.relpath(
        pkg_resources.resource_filename(
            "cellprofiler_core", os.path.join("data", "images", filename)
        )
    )

    # With this specific relative path we are probably building the documentation
    # in sphinx The path separator used by sphinx is "/" on all platforms.
    if relpath == os.path.join("..", "cellprofiler_core", "data", "images", filename):
        return "../images/{}".format(filename)

    # Otherwise, if you're rendering in the GUI, relative paths are fine
    # Note: the HTML renderer requires to paths to use '/' so we replace
    # the windows default '\\' here
    return relpath.replace("\\", "/")


def generate_presigned_url(url):
    """
    Generate a presigned URL, if necessary (e.g., s3).

    :param url: An unsigned URL.
    :return: The presigned URL.
    """
    if url.startswith("s3"):
        client = boto3.client("s3")

        bucket_name, filename = (
            re.compile("s3://([\w\d\-.]+)/(.*)").search(url).groups()
        )

        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": filename.replace("+", " ")},
        )

    return url
