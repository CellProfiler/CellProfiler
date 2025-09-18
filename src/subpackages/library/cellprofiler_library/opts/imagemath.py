from enum import Enum

class Operator(str, Enum):
    ADD = "Add"
    SUBTRACT = "Subtract"
    DIFFERENCE = "Absolute Difference"
    MULTIPLY = "Multiply"
    DIVIDE = "Divide"
    AVERAGE = "Average"
    MINIMUM = "Minimum"
    MAXIMUM = "Maximum"
    STDEV = "Standard Deviation"
    INVERT = "Invert"
    COMPLEMENT = "Complement"
    LOG_TRANSFORM_LEGACY = "Log transform (legacy)"
    LOG_TRANSFORM = "Log transform (base 2)"
    NONE = "None"
    OR = "Or"
    AND = "And"
    NOT = "Not"
    EQUALS = "Equals"


class Operand(str, Enum):
    IMAGE = "Image"
    MEASUREMENT = "Measurement"