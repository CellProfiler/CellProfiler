import os

import numpy
import skimage

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
        mask = properties[0].intensity_image
        
        if file_format.casefold() == "png":
            skimage.io.imsave(
                label_save_filename,
                skimage.img_as_ubyte(mask),
                check_contrast=False
            )
        elif file_format.casefold() == "tiff8":
            skimage.io.imsave(
                label_save_filename,
                skimage.img_as_ubyte(mask),
                compression=(8,6),
                check_contrast=False,
            )
        elif file_format.casefold() == "tiff16":
            skimage.io.imsave(
                label_save_filename,
                skimage.img_as_uint(mask),
                compression=(8,6),
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
            skimage.io.imsave(
                label_save_filename,
                skimage.img_as_ubyte(mask),
                compression=(8, 6),
                check_contrast=False,
            )
        elif file_format.casefold() == "tiff16":
            skimage.io.imsave(
                label_save_filename,
                skimage.img_as_uint(mask),
                compression=(8, 6),
                check_contrast=False,
            )
        else:
            raise ValueError(f"{file_format} not in 'png', 'tiff8', or 'tiff16'")
        
    return filenames