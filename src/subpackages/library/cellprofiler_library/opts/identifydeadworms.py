C_LOCATION = "Location"

FTR_CENTER_X = "Center_X"
M_LOCATION_CENTER_X = f"{C_LOCATION}_{FTR_CENTER_X}"

FTR_CENTER_Y = "Center_Y"
M_LOCATION_CENTER_Y = f"{C_LOCATION}_{FTR_CENTER_Y}"

C_WORMS = "Worm"
F_ANGLE = "Angle"
M_ANGLE = f"{C_WORMS}_{F_ANGLE}"

C_NUMBER = "Number"
FTR_OBJECT_NUMBER = "Object_Number"
M_NUMBER_OBJECT_NUMBER = f"{C_NUMBER}_{FTR_OBJECT_NUMBER}"

C_COUNT = "Count"

class TemplateMeasurementFormat(str):
    FF_COUNT = f"{C_COUNT}_%s"
