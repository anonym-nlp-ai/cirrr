from enum import Enum

class DataExceptionType(str, Enum):
    ENCODING_MISMATCH = "exception.encoding.mismatch"
    CONTEXT_MISMATCH = "exception.context.mismatch"


class ProcessedStatus(str, Enum):
    # "F"	Failed to process, unknown exception, consult system administrator
    F = "F"
    # "N": document not found, do not proceed but log status.
    N = "N"
    # "X" mismatch of information, do not proceed but log status
    X = "X"
    # "Y": checks are valid, ok to proceed.
    Y = "Y"