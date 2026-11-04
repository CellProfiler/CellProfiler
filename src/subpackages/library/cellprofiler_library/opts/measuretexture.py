from enum import Enum

class HaralickFeatures(str, Enum):    
    AngularSecondMoment = "AngularSecondMoment",
    Contrast = "Contrast",
    Correlation = "Correlation",
    Variance = "Variance",
    InverseDifferenceMoment = "InverseDifferenceMoment",
    SumAverage = "SumAverage",
    SumVariance = "SumVariance",
    SumEntropy = "SumEntropy",
    Entropy = "Entropy",
    DifferenceVariance = "DifferenceVariance",
    DifferenceEntropy = "DifferenceEntropy",
    InfoMeas1 = "InfoMeas1",
    InfoMeas2 = "InfoMeas2"

class MeasurementTarget(str, Enum):
    IMAGES = "Images",
    OBJECTS = "Objects"
    BOTH = "Both"
    
TEXTURE = "Texture"
F_HARALICK = [i for i in HaralickFeatures]