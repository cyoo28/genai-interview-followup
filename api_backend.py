from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class Request(BaseModel):
    question: str
    answer: str
    role: Optional[str] = None
    interview_type: Optional[list[str]] = None

@app.post("/interview/generate-followups")
def generate_followups(request: Request):

    return {
        "result": "success",
        "message": "Follow-up question generated.",
        "data": "temp"
    }