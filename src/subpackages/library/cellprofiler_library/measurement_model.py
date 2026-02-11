from typing import Any, Dict, List
import copy
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from numpy.typing import NDArray

# Constants for relationship measurements
IMAGE_NUMBER = "ImageNumber"
OBJECT_NUMBER = "ObjectNumber"
R_FIRST_OBJECT_NUMBER = f"{OBJECT_NUMBER}_First"
R_SECOND_OBJECT_NUMBER = f"{OBJECT_NUMBER}_Second"

class RelationshipBase:
    """Key for identifying relationship groups."""
    def __init__(self, relationship: str, object_name1: str, object_name2: str):
        self.relationship = relationship
        self.object_name1 = object_name1
        self.object_name2 = object_name2

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RelationshipBase):
            return NotImplemented

        return (
            self.relationship == other.relationship
            and self.object_name1 == other.object_name1
            and self.object_name2 == other.object_name2
        )

    def __hash__(self):
        return hash(
            (self.relationship, self.object_name1, self.object_name2)
        )
    
    def __repr__(self):
        return f"RelationshipBase({self.relationship}, {self.object_name1}, {self.object_name2})"

class Relationship(RelationshipBase):
    def __init__(self, relationship: str, object_name1: str, object_name2: str, object_numbers1: NDArray[np.int_], object_numbers2: NDArray[np.int_]):
        super().__init__(relationship, object_name1, object_name2)
        self.object_numbers1 = object_numbers1
        self.object_numbers2 = object_numbers2

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relationship):
            return NotImplemented

        return super().__eq__(other) and np.array_equal(self.object_numbers1, other.object_numbers1) and np.array_equal(self.object_numbers2, other.object_numbers2)

    def __copy__(self):
        new_instance = Relationship(
            self.relationship,
            self.object_name1,
            self.object_name2,
            self.object_numbers1,
            self.object_numbers2
        )
        new_instance.__dict__.update(self.__dict__)
        return new_instance

    def __deepcopy__(self, memo):
        return Relationship(
            self.relationship,
            self.object_name1,
            self.object_name2,
            self.object_numbers1.copy(),
            self.object_numbers2.copy()
        )
 
    def __getitem__(self, obj_num: str) -> NDArray[np.int_]:
        if obj_num == R_FIRST_OBJECT_NUMBER:
            return self.object_numbers1
        elif obj_num == R_SECOND_OBJECT_NUMBER:
            return self.object_numbers2
        else:
            return NotImplemented


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
    relationships: List[Relationship] = Field(default_factory=list, alias="Relationship")

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
        relationship: str,
        object_name1: str,
        object_name2: str,
        object_numbers1: NDArray[np.int_],
        object_numbers2: NDArray[np.int_],
    ):
        """Helper to add a relate measurement.
        
        Args:
            relationship: Name of relationship (e.g., 'Parent')
            object_name1: Name of first object type
            object_name2: Name of second object type
            object_numbers1: Array of indices for first objects
            object_numbers2: Array of indices for second objects (must match length of object_numbers1)
        """
        if len(object_numbers1) == 0:
            return

        # Check if this relationship group already exists
        target_item = None
        for item in self.relationships:
            if (
                item.relationship == relationship
                and item.object_name1 == object_name1
                and item.object_name2 == object_name2
            ):
                target_item = item
                break

        if target_item is None:
            target_item = Relationship(
                relationship,
                object_name1,
                object_name2,
                np.array([], dtype=np.int32),
                np.array([], dtype=np.int32),
            )
            self.relationships.append(target_item)

        # Append data
        target_item.object_numbers1 = np.concatenate(
            (target_item.object_numbers1, object_numbers1)
        )
        target_item.object_numbers2 = np.concatenate(
            (target_item.object_numbers2, object_numbers2)
        )

    def get_relationship_groups(self) -> List[RelationshipBase]:
        """Return the keys of each of the relationship groupings."""
        return [
            RelationshipBase(
                item.relationship,
                item.object_name1,
                item.object_name2,
            )
            for item in self.relationships
        ]

    def get_relationships(
        self,
        relationship: str,
        object_name1: str,
        object_name2: str,
    ) -> NDArray:
        """Get the relationships recorded by a particular module.

        Returns a recarray with fields:
        R_FIRST_OBJECT_NUMBER, R_SECOND_OBJECT_NUMBER
        """
        features = (
            R_FIRST_OBJECT_NUMBER,
            R_SECOND_OBJECT_NUMBER,
        )
        dt = np.dtype([(feature, np.int32, ()) for feature in features])

        # Find the relationship group
        target_item = None
        for item in self.relationships:
            if (
                item.relationship == relationship
                and item.object_name1 == object_name1
                and item.object_name2 == object_name2
            ):
                target_item = item
                break

        if target_item is None:
            return np.zeros(0, dt).view(np.recarray)

        n_records = len(target_item.object_numbers1)
        if n_records == 0:
            return np.zeros(0, dt).view(np.recarray)

        temp = np.zeros(n_records, dt)
        for feature in features:
            temp[feature] = target_item[feature]
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
                 # Check if this relationship group already exists
                if (
                    item.relationship == other_item.relationship
                    and item.object_name1 == other_item.object_name1
                    and item.object_name2 == other_item.object_name2
                ):
                    match_index = i
                    break

            
            if match_index != -1:
                # Merge into existing item
                target = new_relationships[match_index]
                target.object_numbers1 = np.concatenate(
                    (target.object_numbers1, other_item.object_numbers1)
                )
                target.object_numbers2 = np.concatenate(
                    (target.object_numbers2, other_item.object_numbers2)
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
    
