from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ValidationError
from typing import Optional
from openai import OpenAI
import json

# Initialize FastAPI app
app = FastAPI()
# Initialize OpenAI client (uses credentials configured in environment)
client = OpenAI()

# Schema for incoming interview data
class Request(BaseModel):
    question: str                                   # Interviewer's original question
    answer: str                                     # Candidate's response
    role: Optional[str] = None                      # (Optional) Target role
    interview_type: Optional[list[str]] = None      # (Optional) Interview type

# Schema for a single follow-up question
class FollowUp(BaseModel):
    question: str
    rationale: str

# Schema for the full response
class FollowUpResponse(BaseModel):
    followups: list[FollowUp]

# Model to be used for generating follow-up questions
gpt_model = "gpt-5-mini"

# System-level instructions for the model to ensure safe, professional outputs
system_prompt = """
    You are an interviewer assistant. Generate 1–3 concise follow-up questions, that are each less than 50 words, based only on the candidate's answer and the original question. 
    Use role and interview type for context if provided. Include a 1-sentence rationale for each question. 
    Keep questions neutral, professional, and safe. Avoid sensitive personal topics. Do not give advice or opinions. 
    Output must be strict JSON with the same structure as this example: {"followups":[{"question":"...","rationale":"..."}, ...]}
    """

@app.post("/interview/generate-followups")
def generate_followups(request: Request):
    """
    API backend to generate interview follow-up questions.

    Input: Request object containing original question, answer, role, and interview type.
    Output: JSON with generated follow-up questions and rationales.
    """
    # Format optional values; default to "n/a" if not provided
    interview_type = ", ".join(request.interview_type) if request.interview_type else "n/a"
    role = request.role if request.role else "n/a"
    # Send request to OpenAI model with model parameters
    response = client.responses.create(
        model=gpt_model,
        reasoning={"effort": "medium"},
        max_output_tokens=1000,
        instructions=system_prompt,
        input=f"""
            Original Question: {request.question}
            Candidate Answer: {request.answer} 
            Role: {role}
            Interview type: {interview_type}
            """
    )
    # Raise error if model output is incomplete
    if response.status == "incomplete":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= {
                "result": "failure",
                "message": "Model output incomplete.",
                "data": response.incomplete_details.reason
                }
        )
    # Get output text from model
    output_text = response.output_text or ""
    # Raise error if output is empty
    if not output_text:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "result": "failure",
                "message": "Model returned empty output."}
        )

    try:
        # Attempt to parse the model's JSON output and extract "followups" list
        followups = FollowUpResponse.model_validate_json(output_text)
        #followups = FollowUpResponse.parse_raw(response.output_text)["followups"]
    except (json.JSONDecodeError, KeyError, ValidationError):
        # Handle cases where output is not valid JSON or missing expected keys
        # Return HTTP 500 to indicate server-side failure and provide raw output for debugging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail= {
                "result": "failure",
                "message": "Follow-up question failed.",
                "data": output_text
                }
        )
    # Successful parsing; return follow-up questions to client
    return {
        "result": "success",
        "message": "Follow-up question generated.",
        "data": followups.model_dump()
    }