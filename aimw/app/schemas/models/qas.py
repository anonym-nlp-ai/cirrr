from pydantic import BaseModel, ConfigDict, Field, field_validator


class QASet(BaseModel):

    context: str
    qa_set: dict = Field(..., description="Generated QAG")
    model_config = ConfigDict(from_attributes=True)

    @field_validator("context")
    def context_must_be_non_empty(cls, v):
        if not v:
            raise ValueError("Context must not be empty.")
        return v
