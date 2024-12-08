from enum import Enum

# Transformations params
QUILTING_PATCH_SIZE = 5
TVM_WEIGHT = 0.03
PIXEL_DROP_RATE = 0.5
TVM_METHOD = 'chambolle'


class DefenseType(Enum):
    RAW = "raw"
    TVM = 'tvm'
    QUILTING = 'quilting'
    ENSEMBLE_TRAINING = 'ensemble_training'
    JPEG = 'jpeg'
    QUANTIZATION = 'quantize'
    NRP = 'nrp'
    MEDIAN_SMOOTH = 'ms'

    @classmethod
    def has_value(cls, value):
        return (any(value == item.value for item in cls))

    def __str__(self):
        return str(self.value)