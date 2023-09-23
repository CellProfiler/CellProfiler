from ..functions.file_processing import save_object_image_crops, save_object_masks

def savecroppedobjects(
    input_objects,
    save_dir,
    export_as="masks",
    input_image=None,
    file_format="tiff8",
    nested_save=False,
    save_names={"input_filename": None, "input_objects_name": None},
    volumetric=False
    ):
    if export_as.casefold() in ("image", "images"):
        filenames = save_object_image_crops(
            input_image=input_image,
            input_objects=input_objects,
            save_dir=save_dir,
            file_format=file_format,
            nested_save=nested_save,
            save_names=save_names,
            volumetric=volumetric
        )
    elif export_as.casefold() in ("mask", "masks"):
        filenames = save_object_masks(
            input_objects=input_objects,
            save_dir=save_dir,
            file_format=file_format,
            nested_save=nested_save,
            save_names=save_names,
            )
    return filenames
