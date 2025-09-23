import json
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from api_backend import app

client = TestClient(app)

# A valid request, containing all fields
complete_request = {
    "question": "Can you describe a project where you implemented AI or machine learning to solve a real-world problem?",
    "answer": """
        I’ve been working on developing a consumer-facing chatbot product that leverages large language models. 
        My focus has been on prompt engineering, designing structured prompts and leveraging tools like RAG to ensure the AI outputs are accurate and aligned with user expectations. 
        I implemented proper guardrails using AWS Bedrock and Vertex AI, ensuring responses are filtered for unsafe content while maintaining a natural conversational flow. 
    """,
    "role": "AI Engineer",
    "interview_type": ["Technical", "Screening"]
}

# A valid request, missing optional fields (no role or interview_type)
minimal_request = {
    "question": "Can you describe a project where you implemented AI or machine learning to solve a real-world problem?",
    "answer": """
        I’ve been working on developing a consumer-facing chatbot product that leverages large language models. 
        My focus has been on prompt engineering, designing structured prompts and leveraging tools like RAG to ensure the AI outputs are accurate and aligned with user expectations. 
        I implemented proper guardrails using AWS Bedrock and Vertex AI, ensuring responses are filtered for unsafe content while maintaining a natural conversational flow. 
    """
}

# An invalid request, missing required fields (no question)
missing_question_request = {
    "answer": """
        I’ve been working on developing a consumer-facing chatbot product that leverages large language models. 
        My focus has been on prompt engineering, designing structured prompts and leveraging tools like RAG to ensure the AI outputs are accurate and aligned with user expectations. 
        I implemented proper guardrails using AWS Bedrock and Vertex AI, ensuring responses are filtered for unsafe content while maintaining a natural conversational flow. 
    """,
    "role": "AI Engineer",
    "interview_type": ["Technical", "Screening"]
}

# An invalid request, having an invalid field type (interview_type should be a list of strings)
invalid_type_request = {
    "question": "Can you describe a project where you implemented AI or machine learning to solve a real-world problem?",
    "answer": """
        I’ve been working on developing a consumer-facing chatbot product that leverages large language models. 
        My focus has been on prompt engineering, designing structured prompts and leveraging tools like RAG to ensure the AI outputs are accurate and aligned with user expectations. 
        I implemented proper guardrails using AWS Bedrock and Vertex AI, ensuring responses are filtered for unsafe content while maintaining a natural conversational flow. 
    """,
    "role": "AI Engineer",
    "interview_type": "Technical"
}

# Mock response from OpenAI (actually pulled from backend)
mock_output_text = {
    "followups": [
        {
            "question": "Can you describe the RAG pipeline you implemented—retriever type, embedding model, vector store, and how you integrated retrieval results into prompts?",
            "rationale": "To assess the technical choices and how retrieval data was incorporated into prompt design for accuracy."
        },
        {
            "question": "How did you measure and validate the chatbot's accuracy, hallucination rate, and conversational quality—what metrics and evaluation procedures did you use?",
            "rationale": "To understand how the model's real-world performance and alignment were quantitatively and qualitatively evaluated."
        },
        {
            "question": "What specific guardrail mechanisms did you implement within AWS Bedrock and Vertex AI (e.g., classifiers, filters, moderated prompts), and how do they handle false positives or blocked useful content?",
            "rationale": "To probe the safety implementation details and how trade-offs between safety and usability were managed."
        }
    ]
}

# Test 1: Successful follow-up generation
def test_success():
    # Mock the OpenAI client to simulate a successful response
    with patch("api_backend.client.responses.create") as mock_create:
        mock_response = MagicMock()
        mock_response.status = "succeeded"
        mock_response.output_text = json.dumps(mock_output_text)
        mock_create.return_value = mock_response
        # Make a request with a complete, valid request
        response = client.post("/interview/generate-followups", json=complete_request)
        # Check for HTTP 200 OK
        assert response.status_code == 200
        data = response.json()
        # Check that returned result indicates success
        assert data["result"] == "success"
        # Check that there is at least one follow-up question returned
        assert len(data["data"]["followups"]) >= 1

# Test 2: Missing required field (question)
def test_missing_required_field():
    # Make a request missing the required 'question' field
    response = client.post("/interview/generate-followups", json=missing_question_request)
    # Check that FastAPI rejects request with 442 validation error
    assert response.status_code == 422

# Test 3: Invalid field type (interview_type should be a list of strings)
def test_invalid_field_type():
    # Make a request with a wrong data type for 'interview_type'
    response = client.post("/interview/generate-followups", json=invalid_type_request)
    # Check that FastAPI rejects request with 442 validation error
    assert response.status_code == 422

# Test 4: Valid request with missing optional fields
def test_missing_optional_fields():
    # Mock the OpenAI client to simulate a successful response
    with patch("api_backend.client.responses.create") as mock_create:
        mock_response = MagicMock()
        mock_response.status = "succeeded"
        mock_response.output_text = json.dumps(mock_output_text)
        mock_create.return_value = mock_response
        # Make a request with a minimal, valid request
        response = client.post("/interview/generate-followups", json=minimal_request)
        # Check for HTTP 200 OK
        assert response.status_code == 200
        data = response.json()
        # Check that returned result indicates success
        assert data["result"] == "success"
        # Check that there is at least one follow-up question returned
        assert len(data["data"]["followups"]) >= 1