from ..functions.image_processing import morphological_skeleton_2d, morphological_skeleton_3d

def morphologicalskeleton(image, volumetric):
    if volumetric: 
        return morphological_skeleton_3d(image)
    else:
        return morphological_skeleton_2d(image)

