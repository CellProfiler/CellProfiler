import os
import re

import boto3
import pkg_resources


def image_resource(filename):
    try:
        abspath = os.path.abspath(
            pkg_resources.resource_filename(
                "cellprofiler", os.path.join("data", "images", filename)
            )
        )
        return abspath.replace("\\", "/")
    except ModuleNotFoundError:
        # CellProfiler is not installed so the assets are missing.
        # In theory an icon should never be called without the GUI anyway
        print("CellProfiler image assets were not found")
    return ""


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
