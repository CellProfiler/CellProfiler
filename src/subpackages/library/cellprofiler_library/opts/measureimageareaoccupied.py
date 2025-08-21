from enum import Enum


C_AREA_OCCUPIED = "AreaOccupied"

# # Measurement feature name format for the AreaOccupied/VolumeOccupied measurement

class MeasurementType(str, Enum):
    AREA_OCCUPIED = "AreaOccupied"
    VOLUME_OCCUPIED = "VolumeOccupied"
    PERIMETER = "Perimeter"
    SURFACE_AREA = "SurfaceArea"
    TOTAL_AREA = "TotalArea"
    TOTAL_VOLUME = "TotalVolume"
    
class Target(str, Enum):
    BINARY_IMAGE = "Binary Image"
    OBJECTS = "Objects"
    BOTH = "Both"