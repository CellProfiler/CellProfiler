from enum import Enum

# Catrgoty string for the AreaOccupied/VolumeOccupied measurement
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

class TemplateMeasurementFormat(str):
    VOLUME_OCCUPIED_FORMAT = f"{C_AREA_OCCUPIED}_{MeasurementType.VOLUME_OCCUPIED.value}_%s"
    SURFACE_AREA_FORMAT = f"{C_AREA_OCCUPIED}_{MeasurementType.SURFACE_AREA.value}_%s"
    TOTAL_VOLUME_FORMAT = f"{C_AREA_OCCUPIED}_{MeasurementType.TOTAL_VOLUME.value}_%s"
    AREA_OCCUPIED_FORMAT = f"{C_AREA_OCCUPIED}_{MeasurementType.AREA_OCCUPIED.value}_%s"
    PERIMETER_FORMAT = f"{C_AREA_OCCUPIED}_{MeasurementType.PERIMETER.value}_%s"
    TOTAL_AREA_FORMAT = f"{C_AREA_OCCUPIED}_{MeasurementType.TOTAL_AREA.value}_%s"