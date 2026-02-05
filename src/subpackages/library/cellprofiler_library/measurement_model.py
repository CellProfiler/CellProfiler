from typing import Any, Dict, List
import copy
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from numpy.typing import NDArray
class RelationshipKey:
    """Key for identifying relationship groups."""
    def __init__(self, module_number, relationship, object_name1, object_name2):
        self.module_number = module_number
        self.relationship = relationship
        self.object_name1 = object_name1
        self.object_name2 = object_name2

    def __eq__(self, other):
        return (
            self.module_number == other.module_number
            and self.relationship == other.relationship
            and self.object_name1 == other.object_name1
            and self.object_name2 == other.object_name2
        )

    def __hash__(self):
        return hash(
            (self.module_number, self.relationship, self.object_name1, self.object_name2)
        )
    
    def __repr__(self):
        return f"RelationshipKey({self.module_number}, {self.relationship}, {self.object_name1}, {self.object_name2})"


# Constants for relationship measurements
IMAGE_NUMBER = "ImageNumber"
OBJECT_NUMBER = "ObjectNumber"
R_FIRST_IMAGE_NUMBER = f"{IMAGE_NUMBER}_First"
R_FIRST_OBJECT_NUMBER = f"{OBJECT_NUMBER}_First"
R_SECOND_IMAGE_NUMBER = f"{IMAGE_NUMBER}_Second"
R_SECOND_OBJECT_NUMBER = f"{OBJECT_NUMBER}_Second"


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

    def add_relate_measurement(
        self,
        module_number: int,
        relationship: str,
        object_name1: str,
        object_name2: str,
        image_numbers1: NDArray[np.int_],
        object_numbers1: NDArray[np.int_],
        image_numbers2: NDArray[np.int_],
        object_numbers2: NDArray[np.int_],
    ):
        """Helper to add a relate measurement."""
        if len(image_numbers1) == 0:
            return

        # Check if this relationship group already exists
        # We need to find if there is an existing entry for this key
        # In Core, relationships are stored in HDF5. Here we store in a list.
        # However, to mirror Core's add_relate_measurement which APPENDS data,
        # we can just append a new dict or merge with existing one.
        # Merging is better to keep the list clean with one entry per key,
        # matching Core's structure where each key points to a dataset that grows.

        # Structure of stored relationship item:
        # {
        #   "module_number": ...,
        #   "relationship": ...,
        #   "object_name1": ...,
        #   "object_name2": ...,
        #   R_FIRST_IMAGE_NUMBER: array,
        #   ...
        # }

        target_item = None
        for item in self.relationships:
            if (
                item["module_number"] == module_number
                and item["relationship"] == relationship
                and item["object_name1"] == object_name1
                and item["object_name2"] == object_name2
            ):
                target_item = item
                break

        if target_item is None:
            target_item = {
                "module_number": module_number,
                "relationship": relationship,
                "object_name1": object_name1,
                "object_name2": object_name2,
                R_FIRST_IMAGE_NUMBER: np.array([], dtype=np.int32),
                R_FIRST_OBJECT_NUMBER: np.array([], dtype=np.int32),
                R_SECOND_IMAGE_NUMBER: np.array([], dtype=np.int32),
                R_SECOND_OBJECT_NUMBER: np.array([], dtype=np.int32),
            }
            self.relationships.append(target_item)

        # Append data
        target_item[R_FIRST_IMAGE_NUMBER] = np.concatenate(
            (target_item[R_FIRST_IMAGE_NUMBER], image_numbers1)
        )
        target_item[R_FIRST_OBJECT_NUMBER] = np.concatenate(
            (target_item[R_FIRST_OBJECT_NUMBER], object_numbers1)
        )
        target_item[R_SECOND_IMAGE_NUMBER] = np.concatenate(
            (target_item[R_SECOND_IMAGE_NUMBER], image_numbers2)
        )
        target_item[R_SECOND_OBJECT_NUMBER] = np.concatenate(
            (target_item[R_SECOND_OBJECT_NUMBER], object_numbers2)
        )

    def get_relationship_groups(self) -> List[RelationshipKey]:
        """Return the keys of each of the relationship groupings."""
        return [
            RelationshipKey(
                item["module_number"],
                item["relationship"],
                item["object_name1"],
                item["object_name2"],
            )
            for item in self.relationships
        ]

    def get_relationships(
        self,
        module_number: int,
        relationship: str,
        object_name1: str,
        object_name2: str,
        image_numbers: NDArray[np.int_] = None,
    ) -> NDArray:
        """Get the relationships recorded by a particular module.

        Returns a recarray with fields:
        R_FIRST_IMAGE_NUMBER, R_SECOND_IMAGE_NUMBER, R_FIRST_OBJECT_NUMBER,
        R_SECOND_OBJECT_NUMBER
        """
        features = (
            R_FIRST_IMAGE_NUMBER,
            R_FIRST_OBJECT_NUMBER,
            R_SECOND_IMAGE_NUMBER,
            R_SECOND_OBJECT_NUMBER,
        )
        dt = np.dtype([(feature, np.int32, ()) for feature in features])

        # Find the relationship group
        target_item = None
        for item in self.relationships:
            if (
                item["module_number"] == module_number
                and item["relationship"] == relationship
                and item["object_name1"] == object_name1
                and item["object_name2"] == object_name2
            ):
                target_item = item
                break

        if target_item is None:
            return np.zeros(0, dt).view(np.recarray)

        n_records = len(target_item[R_FIRST_IMAGE_NUMBER])
        if n_records == 0:
            return np.zeros(0, dt).view(np.recarray)

        if image_numbers is None:
            temp = np.zeros(n_records, dt)
            for feature in features:
                temp[feature] = target_item[feature]
            return temp.view(np.recarray)
        else:
            image_numbers = np.atleast_1d(image_numbers)
            
            # Create a mask for records matching any of the image numbers
            # Check both first and second image numbers
            mask = np.zeros(n_records, bool)
            
            # Simple implementation for filtering
            # Optimized for small datasets typical in stateless context
            # (LibraryMeasurements usually holds 1 image set, so image_numbers usually match)
            
            img1 = target_item[R_FIRST_IMAGE_NUMBER]
            img2 = target_item[R_SECOND_IMAGE_NUMBER]
            
            for img_num in image_numbers:
                mask |= (img1 == img_num) | (img2 == img_num)
                
            n_kept = np.sum(mask)
            temp = np.zeros(n_kept, dt)
            
            for feature in features:
                temp[feature] = target_item[feature][mask]
                
            return temp.view(np.recarray)

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

    def merge(self, other: 'LibraryMeasurements') -> 'LibraryMeasurements':
        """
        Merge this LibraryMeasurements object with another one, returning a new instance.
        
        Collision resolution:
        - Image measurements: 'other' overwrites 'self'
        - Experiment measurements: 'other' overwrites 'self'
        - Object measurements: Merged by object name. Within an object, 'other' features overwrite 'self'.
        - Relationships: Merged by key (concatenated)
        
        Args:
            other: Another LibraryMeasurements instance
            
        Returns:
            A new LibraryMeasurements instance containing merged data
        """
        # Create deep copies of dictionaries to avoid side effects
        new_image = self.image.copy()
        new_image.update(other.image)
        
        new_experiment = self.experiment.copy()
        new_experiment.update(other.experiment)
        
        new_objects = {}
        # Merge objects
        all_objects = set(self.objects.keys()) | set(other.objects.keys())
        for obj_name in all_objects:
            obj_measurements = self.objects.get(obj_name, {}).copy()
            obj_measurements.update(other.objects.get(obj_name, {}))
            new_objects[obj_name] = obj_measurements
            
        # Merge relationships
        new_relationships = copy.deepcopy(self.relationships)
        
        for other_item in other.relationships:
            # Check if matching item exists
            match_index = -1
            for i, item in enumerate(new_relationships):
                 if (
                    item["module_number"] == other_item["module_number"]
                    and item["relationship"] == other_item["relationship"]
                    and item["object_name1"] == other_item["object_name1"]
                    and item["object_name2"] == other_item["object_name2"]
                ):
                    match_index = i
                    break
            
            if match_index != -1:
                # Merge into existing item
                target = new_relationships[match_index]
                target[R_FIRST_IMAGE_NUMBER] = np.concatenate(
                    (target[R_FIRST_IMAGE_NUMBER], other_item[R_FIRST_IMAGE_NUMBER])
                )
                target[R_FIRST_OBJECT_NUMBER] = np.concatenate(
                    (target[R_FIRST_OBJECT_NUMBER], other_item[R_FIRST_OBJECT_NUMBER])
                )
                target[R_SECOND_IMAGE_NUMBER] = np.concatenate(
                    (target[R_SECOND_IMAGE_NUMBER], other_item[R_SECOND_IMAGE_NUMBER])
                )
                target[R_SECOND_OBJECT_NUMBER] = np.concatenate(
                    (target[R_SECOND_OBJECT_NUMBER], other_item[R_SECOND_OBJECT_NUMBER])
                )
            else:
                # Append new item (deep copy to be safe)
                new_relationships.append(copy.deepcopy(other_item))
        
        return LibraryMeasurements(
            image=new_image,
            objects=new_objects,
            experiment=new_experiment,
            relationships=new_relationships
        )
    
    @staticmethod
    def from_cellprofiler_measurements(measurements) -> 'LibraryMeasurements':
        """
        Create a LibraryMeasurements instance from a CellProfiler Measurements instance.
        
        Args:
            measurements: A CellProfiler Measurements instance
            
        Returns:
            A LibraryMeasurements instance
        """
        image = {}
        objects = {}
        experiment = {}
        relationships = []
        
        for object_name in measurements.get_object_names():
            objects[object_name] = {}
            for feature in measurements.get_feature_names(object_name):
                objects[object_name][feature] = measurements.get_measurement(object_name, feature)
        
        for feature in measurements.get_feature_names("Image"):
            image[feature] = measurements.get_measurement("Image", feature)
        try:
            for feature in measurements.get_feature_names("Experiment"):
                experiment[feature] = measurements.get_measurement("Experiment", feature)
        except KeyError:
            pass
        
        for relationship in measurements.get_relationship_groups():
            relationships.append({
                "module_number": relationship["module_number"],
                "relationship": relationship["relationship"],
                "object_name1": relationship["object_name1"],
                "object_name2": relationship["object_name2"],
                "image_number1": relationship[R_FIRST_IMAGE_NUMBER],
                "object_number1": relationship[R_FIRST_OBJECT_NUMBER],
                "image_number2": relationship[R_SECOND_IMAGE_NUMBER],
                "object_number2": relationship[R_SECOND_OBJECT_NUMBER],
            })
        
        return LibraryMeasurements(
            image=image,
            objects=objects,
            experiment=experiment,
            relationships=relationships
        )   
