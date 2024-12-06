from enum import Enum


class DefenseType(Enum):
    RAW = "raw"
    TVM = 'tvm'
    QUILTING = 'quilting'
    ENSEMBLE_TRAINING = 'ensemble_training'
    JPEG = 'jpeg'
    QUANTIZATION = 'quantize'
    NRP = 'nrp'

    @classmethod
    def has_value(cls, value):
        return (any(value == item.value for item in cls))

    def __str__(self):
        return str(self.value)