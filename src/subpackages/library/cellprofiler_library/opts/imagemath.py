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

BINARY_OUTPUT_OPS = [Operator.AND, Operator.OR, Operator.NOT, Operator.EQUALS]
UNARY_INPUT_OPS = [Operator.STDEV, Operator.INVERT, Operator.NOT, Operator.LOG_TRANSFORM, Operator.LOG_TRANSFORM_LEGACY, Operator.NONE]
BINARY_INPUT_OPS = [Operator.ADD, Operator.SUBTRACT, Operator.DIFFERENCE, Operator.MULTIPLY, Operator.DIVIDE, Operator.AVERAGE, Operator.MAXIMUM, Operator.MINIMUM, Operator.AND, Operator.OR, Operator.EQUALS]