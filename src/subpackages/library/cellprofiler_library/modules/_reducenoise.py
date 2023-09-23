from ..functions.image_processing import reduce_noise

def reducenoise(image, patch_size, patch_distance, cutoff_distance, channel_axis=None):
    denoised = reduce_noise(
        image,
        patch_size=patch_size,
        patch_distance=patch_distance,
        cutoff_distance=cutoff_distance,
        channel_axis=channel_axis,
    )
    return denoised
