from typing import Optional
from pydantic import BaseModel
from typing import List


class NoteSchema(BaseModel):
    title: Optional[str]
    content: Optional[str]

    class Config:
        schema_extra = {
            "example": {
                "title": "LogRocket.",
                "content": "Logrocket is the most flexible publishing company for technical authors. From editors to payment, the process is too flexible and that's what makes it great.",
            }
        }


class PostStrokeBody(BaseModel):
    strokes: List[List[int]]
