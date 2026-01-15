from typing import Any, Dict, List
from pydantic import BaseModel, Field, ConfigDict

class LibraryMeasurements(BaseModel):
    """
    A Pydantic implementation mirroring CellProfiler's Measurements class functionality.
    
    This class represents measurements for a single image set context, designed to be
    constructed from and serializable to the "Primitive Dictionary" contract.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True, 
        populate_by_name=True
    )

    # Primary data structures matching the MDC contract
    image: Dict[str, Any] = Field(default_factory=dict, alias="Image")
    objects: Dict[str, Dict[str, Any]] = Field(default_factory=dict, alias="Object")
    
    # Additional storage for full Measurements compatibility (e.g. Experiment metadata)
    experiment: Dict[str, Any] = Field(default_factory=dict, alias="Experiment")
    
    # Placeholder for relationships if needed in the future
    relationships: List[Dict[str, Any]] = Field(default_factory=list, alias="Relationship")

    def add_measurement(self, object_name: str, feature_name: str, data: Any):
        """
        Add a measurement. Mirrors Measurements.add_measurement.
        
        Args:
            object_name: Name of the object (e.g., 'Image', 'Nuclei').
            feature_name: Name of the feature (e.g., 'AreaShape_Area').
            data: The value to store (float, string, or numpy array).
        """
        if object_name == "Experiment":
            self.experiment[feature_name] = data
        elif object_name == "Image":
            self.image[feature_name] = data
        else:
            if object_name not in self.objects:
                self.objects[object_name] = {}
            self.objects[object_name][feature_name] = data

    def get_measurement(self, object_name: str, feature_name: str) -> Any:
        """
        Get a measurement. Mirrors Measurements.get_measurement.
        """
        if object_name == "Experiment":
            return self.experiment.get(feature_name)
        elif object_name == "Image":
            return self.image.get(feature_name)
        else:
            return self.objects.get(object_name, {}).get(feature_name)

    def add_image_measurement(self, feature_name: str, data: Any):
        """Helper to add an image measurement."""
        self.add_measurement("Image", feature_name, data)

    def add_experiment_measurement(self, feature_name: str, data: Any):
        """Helper to add an experiment measurement."""
        self.add_measurement("Experiment", feature_name, data)

    def get_experiment_measurement(self, feature_name: str) -> Any:
        """Helper to get an experiment measurement."""
        return self.get_measurement("Experiment", feature_name)
    
    def get_object_names(self) -> List[str]:
        """Return a list of all object names (including Image and Experiment if present)."""
        names = list(self.objects.keys())
        if self.image:
            names = ["Image"] + names
        if self.experiment:
            names = ["Experiment"] + names
        return names

    def get_feature_names(self, object_name: str) -> List[str]:
        """Return feature names for a specific object."""
        if object_name == "Experiment":
            return list(self.experiment.keys())
        elif object_name == "Image":
            return list(self.image.keys())
        else:
            return list(self.objects.get(object_name, {}).keys())
            
    def has_feature(self, object_name: str, feature_name: str) -> bool:
        """Check if a feature exists."""
        if object_name == "Experiment":
            return feature_name in self.experiment
        elif object_name == "Image":
            return feature_name in self.image
        else:
            return feature_name in self.objects.get(object_name, {})

    def __getitem__(self, key):
        """Support dict-style access: m['Image', 'Count_Cells']"""
        if isinstance(key, tuple):
             assert len(key) == 2
             return self.get_measurement(*key)
        raise KeyError("Invalid key format. Expected (Object, Feature)")
        
    def __setitem__(self, key, value):
        """Support dict-style assignment: m['Image', 'Count_Cells'] = 5"""
        if isinstance(key, tuple):
             assert len(key) == 2
             self.add_measurement(*key, value)
        else:
             raise KeyError("Invalid key format. Expected (Object, Feature)")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to the primitive dictionary format required by the refactoring contract.
        """
        return self.model_dump(by_alias=True, exclude_none=True)
