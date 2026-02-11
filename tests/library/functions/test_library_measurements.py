
import numpy as np
import pytest
import copy
from cellprofiler_library.measurement_model import (
    RelationshipBase,
    Relationship,
    LibraryMeasurements,
    R_FIRST_OBJECT_NUMBER,
    R_SECOND_OBJECT_NUMBER,
    IMAGE_NUMBER,
    OBJECT_NUMBER
)

class TestRelationshipBase:
    def test_init(self):
        rb = RelationshipBase("Parent", "Nuclei", "Cells")
        assert rb.relationship == "Parent"
        assert rb.object_name1 == "Nuclei"
        assert rb.object_name2 == "Cells"

    def test_eq(self):
        rb1 = RelationshipBase("Parent", "Nuclei", "Cells")
        rb2 = RelationshipBase("Parent", "Nuclei", "Cells")
        rb3 = RelationshipBase("Child", "Nuclei", "Cells")
        rb4 = RelationshipBase("Parent", "Nuclei", "Cytoplasm")
        
        assert rb1 == rb2
        assert rb1 != rb3
        assert rb1 != rb4
        assert rb1 != "NotARelationshipBase"

    def test_hash(self):
        rb1 = RelationshipBase("Parent", "Nuclei", "Cells")
        rb2 = RelationshipBase("Parent", "Nuclei", "Cells")
        assert hash(rb1) == hash(rb2)
        
        # Verify hash consistency
        d = {rb1: "value"}
        assert d[rb2] == "value"

    def test_repr(self):
        rb = RelationshipBase("Parent", "Nuclei", "Cells")
        assert repr(rb) == "RelationshipBase(Parent, Nuclei, Cells)"


class TestRelationship:
    def test_init(self):
        r = Relationship("Parent", "Nuclei", "Cells", np.array([1]), np.array([2]))
        assert r.relationship == "Parent"
        assert r.object_name1 == "Nuclei"
        assert r.object_name2 == "Cells"
        np.testing.assert_array_equal(r.object_numbers1, np.array([1]))
        np.testing.assert_array_equal(r.object_numbers2, np.array([2]))

    def test_eq(self):
        r1 = Relationship("Parent", "Nuclei", "Cells", np.array([1]), np.array([2]))
        r2 = Relationship("Parent", "Nuclei", "Cells", np.array([1]), np.array([2]))
        r3 = Relationship("Child", "Nuclei", "Cells", np.array([1]), np.array([2]))
        
        assert r1 == r2
        assert r1 != r3
        assert r1 != "NotARelationship"

    def test_neq_relationship_with_different_object_numbers(self):
        r1 = Relationship("Parent", "Nuclei", "Cells", np.array([1, 2]), np.array([2, 3]))
        r2 = Relationship("Parent", "Nuclei", "Cells", np.array([2]), np.array([1]))
        assert r1 != r2  # Equality should consider object numbers and not just the key
        

    def test_copy(self):
        r1 = Relationship("Parent", "Nuclei", "Cells", np.array([1]), np.array([2]))
        r2 = copy.copy(r1)
        assert r1 == r2
        assert r1 is not r2
        # Shallow copy shares arrays
        assert r1.object_numbers1 is r2.object_numbers1

    def test_deepcopy(self):
        r1 = Relationship("Parent", "Nuclei", "Cells", np.array([1]), np.array([2]))
        r2 = copy.deepcopy(r1)
        assert r1 == r2
        assert r1 is not r2
        # Deep copy creates new arrays
        assert r1.object_numbers1 is not r2.object_numbers1
        np.testing.assert_array_equal(r1.object_numbers1, r2.object_numbers1)

    def test_getitem(self):
        r = Relationship("Parent", "Nuclei", "Cells", np.array([1]), np.array([2]))
        np.testing.assert_array_equal(r[R_FIRST_OBJECT_NUMBER], np.array([1]))
        np.testing.assert_array_equal(r[R_SECOND_OBJECT_NUMBER], np.array([2]))
        assert r["Invalid"] == NotImplemented


class TestLibraryMeasurements:
    def test_add_and_get_measurement(self):
        lm = LibraryMeasurements()
        
        # Test object measurement
        lm.add_measurement("Nuclei", "Area", 100)
        assert lm.get_measurement("Nuclei", "Area") == 100
        # Check internal storage structure
        assert lm.objects["Nuclei"]["Area"] == 100

        # Test Image measurement
        lm.add_measurement("Image", "Count_Nuclei", 5)
        assert lm.get_measurement("Image", "Count_Nuclei") == 5
        assert lm.image["Count_Nuclei"] == 5

        # Test Experiment measurement
        lm.add_measurement("Experiment", "Date", "2023-10-27")
        assert lm.get_measurement("Experiment", "Date") == "2023-10-27"
        assert lm.experiment["Date"] == "2023-10-27"

        # Test non-existent measurements
        assert lm.get_measurement("Nuclei", "NonExistent") is None
        assert lm.get_measurement("NonExistentObject", "Area") is None
        assert lm.get_measurement("Image", "NonExistent") is None
        assert lm.get_measurement("Experiment", "NonExistent") is None

    def test_add_helpers(self):
        lm = LibraryMeasurements()
        
        lm.add_image_measurement("Threshold", 0.5)
        assert lm.get_measurement("Image", "Threshold") == 0.5
        
        lm.add_experiment_measurement("Project", "Cell")
        assert lm.get_experiment_measurement("Project") == "Cell"
        assert lm.get_measurement("Experiment", "Project") == "Cell"

    def test_add_relate_measurement_basic(self):
        lm = LibraryMeasurements()
        
        # Add new relationship
        lm.add_relate_measurement(
            "Parent", "Nuclei", "Cells", 
            np.array([1, 2], dtype=np.int32), 
            np.array([10, 20], dtype=np.int32)
        )
        
        assert len(lm.relationships) == 1
        rel = lm.relationships[0]
        assert rel.relationship == "Parent"
        assert rel.object_name1 == "Nuclei"
        assert rel.object_name2 == "Cells"
        np.testing.assert_array_equal(rel.object_numbers1, np.array([1, 2]))
        np.testing.assert_array_equal(rel.object_numbers2, np.array([10, 20]))

    def test_add_relate_measurement_append(self):
        lm = LibraryMeasurements()
        
        # Initial add
        lm.add_relate_measurement(
            "Parent", "Nuclei", "Cells", 
            np.array([1], dtype=np.int32), 
            np.array([10], dtype=np.int32)
        )
        
        # Append to existing
        lm.add_relate_measurement(
            "Parent", "Nuclei", "Cells", 
            np.array([2], dtype=np.int32), 
            np.array([20], dtype=np.int32)
        )
        
        assert len(lm.relationships) == 1
        rel = lm.relationships[0]
        np.testing.assert_array_equal(rel.object_numbers1, np.array([1, 2]))
        np.testing.assert_array_equal(rel.object_numbers2, np.array([10, 20]))

    def test_add_relate_measurement_empty(self):
        lm = LibraryMeasurements()
        lm.add_relate_measurement("Parent", "Nuclei", "Cells", np.array([]), np.array([]))
        assert len(lm.relationships) == 0

    def test_get_relationship_groups(self):
        lm = LibraryMeasurements()
        lm.add_relate_measurement("Parent", "Nuclei", "Cells", np.array([1]), np.array([10]))
        lm.add_relate_measurement("Child", "Nuclei", "Cells", np.array([2]), np.array([20]))
        
        groups = lm.get_relationship_groups()
        assert len(groups) == 2
        assert isinstance(groups[0], RelationshipBase)
        assert groups[0].relationship == "Parent"
        assert groups[1].relationship == "Child"

    def test_get_relationships(self):
        lm = LibraryMeasurements()
        lm.add_relate_measurement("Parent", "Nuclei", "Cells", np.array([1, 2]), np.array([10, 20]))
        
        # Test valid retrieval
        rec = lm.get_relationships("Parent", "Nuclei", "Cells")
        assert isinstance(rec, np.recarray)
        assert len(rec) == 2
        np.testing.assert_array_equal(rec[R_FIRST_OBJECT_NUMBER], np.array([1, 2]))
        np.testing.assert_array_equal(rec[R_SECOND_OBJECT_NUMBER], np.array([10, 20]))
        
        # Test nonexistent relationship
        rec_empty = lm.get_relationships("NonExistent", "A", "B")
        assert len(rec_empty) == 0
        
        # Test empty relationship (manually empty it to test code path if n_records == 0)
        lm.relationships[0].object_numbers1 = np.array([], dtype=np.int32)
        lm.relationships[0].object_numbers2 = np.array([], dtype=np.int32)
        rec_zero = lm.get_relationships("Parent", "Nuclei", "Cells")
        assert len(rec_zero) == 0

    def test_get_object_names(self):
        lm = LibraryMeasurements()
        # Initially empty
        assert lm.get_object_names() == []
        
        lm.add_measurement("Nuclei", "Area", 100)
        assert "Nuclei" in lm.get_object_names()
        
        lm.add_image_measurement("Count", 1)
        names = lm.get_object_names()
        assert names[0] == "Image" # "Image" is prepended
        assert "Nuclei" in names
        
        lm.add_experiment_measurement("Date", "Now")
        names = lm.get_object_names()
        assert names[0] == "Experiment" # "Experiment" is prepended
        assert names[1] == "Image"
        assert "Nuclei" in names

    def test_get_feature_names(self):
        lm = LibraryMeasurements()
        lm.add_measurement("Nuclei", "Area", 100)
        lm.add_image_measurement("Count", 1)
        lm.add_experiment_measurement("Date", "Now")
        
        assert "Area" in lm.get_feature_names("Nuclei")
        assert "Count" in lm.get_feature_names("Image")
        assert "Date" in lm.get_feature_names("Experiment")
        assert lm.get_feature_names("NonExistent") == []

    def test_has_feature(self):
        lm = LibraryMeasurements()
        lm.add_measurement("Nuclei", "Area", 100)
        lm.add_image_measurement("Count", 1)
        lm.add_experiment_measurement("Date", "Now")
        
        assert lm.has_feature("Nuclei", "Area")
        assert not lm.has_feature("Nuclei", "Perimeter")
        assert lm.has_feature("Image", "Count")
        assert not lm.has_feature("Image", "Intensity")
        assert lm.has_feature("Experiment", "Date")
        assert not lm.has_feature("Experiment", "Author")
        assert not lm.has_feature("NonExistent", "Anything")

    def test_dunder_getitem_setitem(self):
        lm = LibraryMeasurements()
        
        # Test setitem
        lm["Nuclei", "Area"] = 100
        assert lm.get_measurement("Nuclei", "Area") == 100
        
        # Test getitem
        assert lm["Nuclei", "Area"] == 100
        
        # Test Invalid Key Format
        with pytest.raises(KeyError, match="Invalid key format"):
            lm["Nuclei"] = 100
            
        with pytest.raises(KeyError, match="Invalid key format"):
            _ = lm["Nuclei"]

    def test_to_dict(self):
        lm = LibraryMeasurements()
        lm.add_measurement("Nuclei", "Area", 100)
        d = lm.to_dict()
        assert isinstance(d, dict)
        assert "Object" in d
        assert "Nuclei" in d["Object"]
        assert d["Object"]["Nuclei"]["Area"] == 100

    def test_merge(self):
        lm1 = LibraryMeasurements()
        lm1.add_measurement("Nuclei", "Area", 100)
        lm1.add_image_measurement("Count", 1)
        lm1.add_relate_measurement("Parent", "Nuclei", "Cells", np.array([1]), np.array([10]))

        lm2 = LibraryMeasurements()
        lm2.add_measurement("Nuclei", "Perimeter", 50) # New feature
        lm2.add_measurement("Nuclei", "Area", 200) # Overwrite existing
        lm2.add_image_measurement("Count", 2) # Overwrite existing
        # Append to existing relationship
        lm2.add_relate_measurement("Parent", "Nuclei", "Cells", np.array([2]), np.array([20]))
        # New relationship
        lm2.add_relate_measurement("Child", "Nuclei", "Cells", np.array([1]), np.array([10]))

        merged = lm1.merge(lm2)
        
        # Verify merges
        assert merged.get_measurement("Nuclei", "Area") == 200
        assert merged.get_measurement("Nuclei", "Perimeter") == 50
        assert merged.get_measurement("Image", "Count") == 2
        
        # Verify relationships
        parent_rels = merged.get_relationships("Parent", "Nuclei", "Cells")
        assert len(parent_rels) == 2
        np.testing.assert_array_equal(parent_rels[R_FIRST_OBJECT_NUMBER], np.array([1, 2]))
        
        child_rels = merged.get_relationships("Child", "Nuclei", "Cells")
        assert len(child_rels) == 1
        
        # Verify original instances are untouched
        assert lm1.get_measurement("Nuclei", "Area") == 100

