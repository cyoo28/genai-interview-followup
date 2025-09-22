from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI

# Initialize FastAPI app
app = FastAPI()
# Initialize OpenAI client (uses credentials configured in environment)
client = OpenAI()

# Request schema for incoming interview data
class Request(BaseModel):
    question: str                                   # Interviewer's original question
    answer: str                                     # Candidate's response
    role: Optional[str] = None                      # (Optional) Target role
    interview_type: Optional[list[str]] = None      # (Optional) Interview type

# Model to be used for generating follow-up questions
gpt_model = "gpt-5-nano"

# System-level instructions for the model to ensure safe, professional outputs
system_prompt = """
    You are an interviewer assistant. Generate concise follow-up questions based on the input provided.
    
    Follow these rules:
    1. Only base the follow-up questions on the candidate's answer and the interviewer's original question. Use role and interview type for context. Do not introduce unrelated topics.
    2. Provide 1–3 concise questions, each 1–2 sentences maximum, under 50 words per question.
    3. After each question, include a very brief rationale (1 sentence) explaining why the question is relevant.
    4. Keep all questions neutral, professional, and safe. Avoid sensitive personal subjects, including (but not limited to) race, gender, religion, and sexual orientation.
    5. Do not give advice, opinions, or feedback.
    6. Make sure questions are clear, grammatically correct, and relevant to the interview context.
    7. Output must be in strict JSON format matching this structure:
    {
        "followups": [
            {
                "question": "Your first follow-up question here",
                "rationale": "Why this question is relevant"
            },
            {
                "question": "Second follow-up question",
                "rationale": "Rationale for second question"
            }
        ]
    }
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
        max_output_tokens=500,
        instructions=system_prompt,
        input=f"""
            Original Question: {request.question}
            Candidate Answer: {request.answer} 
            Role: {role}
            Interview type: {interview_type}
            """
    )
    return {
        "result": "success",
        "message": "Follow-up question generated.",
        "data": "temp"
    }