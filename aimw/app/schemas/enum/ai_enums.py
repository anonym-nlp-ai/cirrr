from enum import Enum


class Role(str, Enum):
    CLASSIFIER = "classifier"
    MODERATOR = "moderator"
    WRITER = "writer"
    WRITER_INITIAL = "writer_initial"
    CURMUDGEON = "curmudgeon"

    def describe(self):
        return self.name, self.value

