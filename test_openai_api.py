import pytest
from openai import OpenAI
from api_backend import call_openai, FollowUpResponse

# Initialize OpenAI client (uses credentials configured in environment)
client = OpenAI()

def test_complete():
    # Define the payload for the test
    payload = {
        "question": "Can you describe a project where you implemented AI or machine learning to solve a real-world problem?",
        "answer": (
            "I’ve been working on developing a consumer-facing chatbot product that leverages large language models. "
            "My focus has been on prompt engineering, designing structured prompts and leveraging tools like RAG to "
            "ensure the AI outputs are accurate and aligned with user expectations. "
            "I implemented proper guardrails using AWS Bedrock and Vertex AI, ensuring responses are filtered for "
            "unsafe content while maintaining a natural conversational flow."
        ),
        "role": "AI Engineer",
        "interview_type": ["Technical", "Screening"]
    }

    # Send request to OpenAI model with model parameters
    response = call_openai(client, payload["question"], payload["answer"], payload["role"], payload["interview_type"])

    # Check that returned result is not empty
    assert response is not None, "OpenAI failed to return response."
    output_text = response.output_text or ""
    assert output_text, "OpenAI returned empty output."

    # Check that there is at least 1 followup
    followups = FollowUpResponse.model_validate_json(output_text)
    assert len(followups.followups) >= 1, "OpenAI has either returned no followups or incorrect format"

    # Check that followups follow the correct format
    for followup in followups.followups:
        assert followup.followup_question, "Followup missing question"
        assert followup.rationale, "Followup missing rationale"