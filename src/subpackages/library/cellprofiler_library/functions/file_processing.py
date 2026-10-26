import os
from pathlib import Path
import warnings

import numpy
import skimage
import tifffile


def save_tiff(filename, pixels, check_contrast=True, **kwargs):
    if isinstance(filename, Path):
        filename = str(filename.resolve())
    if pixels.dtype == bool:
        warnings.warn(
            f"{filename} is a boolean image: setting True to 255 and False to 0. "
            "To silence this warning, please convert the image using img_as_ubyte.",
            stacklevel=2,
        )
        pixels = pixels.astype("uint8") * 255
    if check_contrast and skimage.exposure.is_low_contrast(pixels):
        warnings.warn(f"{filename} is a low contrast image", stacklevel=2)
    if pixels.shape[0] in (3, 4):
        kwargs.setdefault("planarconfig", "SEPARATE")
        rgb = True
    else:
        rgb = pixels.shape[-1] in (3, 4)
    if rgb:
        kwargs.setdefault("photometric", "RGB")
    return tifffile.imwrite(filename, pixels, **kwargs)

def save_object_image_crops(
    input_image,
    input_objects,
    save_dir,
    file_format="tiff8",
    nested_save=False,
    save_names = {"input_filename": None, "input_objects_name": None},
    volumetric=False
    ):
    """
    For a given input_objects array, save crops for each 
    object of the provided input_image.
    """
    # Build save paths
    if nested_save:
        if not save_names["input_filename"] and not save_names["input_objects_name"]:
            raise ValueError("Must provide a save_names['input_filename'] or save_names['input_objects_name'] for nested save.")
        save_path = os.path.join(
            save_dir, 
            save_names["input_filename"] if save_names["input_filename"] else save_names["input_objects_name"],
            )
    else:
        save_path = save_dir

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    unique_labels = numpy.unique(input_objects)

    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]

    labels = input_objects

    if len(input_image.shape) == len(input_objects.shape) + 1 and not volumetric:
        labels = numpy.repeat(
            labels[:, :, numpy.newaxis], input_image.shape[-1], axis=2
        )

    # Construct filename
    save_filename = f"{save_names['input_filename']+'_' if save_names['input_filename'] else ''}{save_names['input_objects_name']+'_' if save_names['input_objects_name'] else ''}"

    save_filenames = []
    
    for label in unique_labels:
        file_extension = "tiff" if "tiff" in file_format else "png"

        label_save_filename = os.path.join(save_path, save_filename + f"{label}.{file_extension}")
        save_filenames.append(label_save_filename)
        mask_in = labels == label
        properties = skimage.measure.regionprops(
                mask_in.astype(int), intensity_image=input_image
            )
        mask = properties[0].image_intensity
        
        if file_format.casefold() == "png":
            skimage.io.imsave(
                label_save_filename,
                skimage.img_as_ubyte(mask),
                check_contrast=False
            )
        elif file_format.casefold() == "tiff8":
            save_tiff(
                label_save_filename,
                skimage.img_as_ubyte(mask),
                compression=8,
                compressionargs={"level": 6},
                check_contrast=False,
            )
        elif file_format.casefold() == "tiff16":
            save_tiff(
                label_save_filename,
                skimage.img_as_uint(mask),
                compression=8,
                compressionargs={"level": 6},
                check_contrast=False,
            )
        else:
            raise ValueError(f"{file_format} not in 'png', 'tiff8', or 'tiff16'")
    
    return save_filenames

def save_object_masks(
    input_objects,
    save_dir,
    file_format="tiff8",
    nested_save=False,
    save_names = {"input_filename": None, "input_objects_name": None},
    ):
    """
    For a given object array, save objects as individual masks
    """
    # Build save paths
    if nested_save:
        if not save_names["input_filename"] and not save_names["input_objects_name"]:
            raise ValueError("Must provide a save_names['input_filename'] or save_names['input_objects_name'] for nested save.")
        save_path = os.path.join(
            save_dir, 
            save_names["input_filename"] if save_names["input_filename"] else save_names["input_objects_name"],
            )
    else:
        save_path = save_dir

    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    unique_labels = numpy.unique(input_objects)

    if unique_labels[0] == 0:
        unique_labels = unique_labels[1:]

    labels = input_objects

    # Construct filename
    save_filename = f"{save_names['input_filename']+'_' if save_names['input_filename'] else ''}{save_names['input_objects_name']+'_' if save_names['input_objects_name'] else ''}"

    filenames = []
    
    for label in unique_labels:
        file_extension = "tiff" if "tiff" in file_format else "png"

        label_save_filename = os.path.join(save_path, save_filename + f"{label}.{file_extension}")

        filenames.append(label_save_filename)

        mask = labels == label
        
        if file_format.casefold() == "png":
                skimage.io.imsave(
                    label_save_filename, 
                    skimage.img_as_ubyte(mask), 
                    check_contrast=False
                )
        elif file_format.casefold() == "tiff8":
            save_tiff(
                label_save_filename,
                skimage.img_as_ubyte(mask),
                compression=8,
                compressionargs={"level": 6},
                check_contrast=False,
            )
        elif file_format.casefold() == "tiff16":
            save_tiff(
                label_save_filename,
                skimage.img_as_uint(mask),
                compression=8,
                compressionargs={"level": 6},
                check_contrast=False,
            )
        else:
            raise ValueError(f"{file_format} not in 'png', 'tiff8', or 'tiff16'")
        
    return filenames
