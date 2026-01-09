from typing import Any, Dict, Optional, Union, List
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, ConfigDict, field_validator, computed_field

class MeasurementTarget(str, Enum):
    IMAGE = "Image"
    OBJECT = "Object"

class Measurement(BaseModel):
    """
    A Pydantic model representing a single measurement in CellProfiler.
    
    This model encapsulates the data required to record a measurement:
    - category: The general category (e.g., "Intensity", "Texture").
    - feature_name: The specific algorithm or feature (e.g., "MeanIntensity").
    - measurement_target: Whether it's an Image or Object measurement.
    - object_name: The name of the objects (required for Object measurements).
    - image_name: The name of the image (optional, often part of the feature name).
    - parameters: A dictionary of parameters used to calculate the measurement.
    - value: The measured value(s).
    """
    
    category: str = Field(..., description="The general category of the measurement.")
    feature_name: str = Field(..., description="The base name of the feature.")
    measurement_target: MeasurementTarget = Field(default=MeasurementTarget.IMAGE, description="Target of the measurement (Image or Object).")
    
    object_name: Optional[str] = Field(default=None, description="Name of the objects (required for Object measurements).")
    image_name: Optional[str] = Field(default=None, description="Name of the input image.")
    
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters used for the measurement calculation.")
    
    value: Union[float, int, np.ndarray, List[float], Any] = Field(..., description="The value of the measurement.")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @field_validator('object_name')
    @classmethod
    def validate_object_name(cls, v: Optional[str], info: Any) -> Optional[str]:
        if info.data.get('measurement_target') == MeasurementTarget.OBJECT and not v:
            raise ValueError('object_name is required for Object measurements')
        return v

    def compute_measurement_name(self) -> str:
        """
        Computes the full measurement name string.
        
        The default implementation joins the category, feature_name, and parameters (values) with underscores.
        Subclasses or users can override this method to implement custom naming logic.
        """
        parts = [self.category, self.feature_name]
        
        # By default, append parameter values in key-sorted order
        # This allows for consistent naming if parameters are used
        for key in sorted(self.parameters.keys()):
            val = self.parameters[key]
            parts.append(str(val))
            
        return "_".join(parts)

    @computed_field
    def name(self) -> str:
        """
        The computed measurement name used by CellProfiler.
        """
        return self.compute_measurement_name()

class MeasurementStatistic(BaseModel):
    """
    A statistic to be displayed in the CellProfiler UI.
    """
    name: str = Field(..., description="Name of the statistic for display.")
    value: str = Field(..., description="Formatted value for display.")

class MeasurementResult(BaseModel):
    """
    A collection of measurements and statistics returned by a measurement module.
    """
    measurements: List[Measurement] = Field(default_factory=list, description="List of measurements to record.")
    statistics: List[MeasurementStatistic] = Field(default_factory=list, description="List of statistics to display.")

    def add_measurement(self, measurement: Measurement) -> None:
        self.measurements.append(measurement)

    def add_statistic(self, name: str, value: str) -> None:
        self.statistics.append(MeasurementStatistic(name=name, value=value))
