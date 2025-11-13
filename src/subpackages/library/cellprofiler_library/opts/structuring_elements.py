from enum import Enum

class StructuringElementShape2D(str, Enum):
    DIAMOND = "Diamond"
    DISK = "Disk"
    SQUARE = "Square"
    STAR = "Star"
    
class StructuringElementShape3D(str, Enum):
    BALL = "Ball"
    CUBE = "Cube"
    OCTAHEDRON = "Octahedron"