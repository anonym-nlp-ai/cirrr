from enum import Enum


class Role(str, Enum):
    CLASSIFIER = "classifier"
    MODERATOR = "moderator"
    WRITER = "writer"
    CURMUDGEON = "curmudgeon"

    def describe(self):
        return self.name, self.value

