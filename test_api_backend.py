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

# Valid response from OpenAI in JSON format (actually pulled from backend)
valid_output_text = {
    "followups": [
        {
            "followup_question": "Can you describe the RAG pipeline you implemented—retriever type, embedding model, vector store, and how you integrated retrieval results into prompts?",
            "rationale": "To assess the technical choices and how retrieval data was incorporated into prompt design for accuracy."
        },
        {
            "followup_question": "How did you measure and validate the chatbot's accuracy, hallucination rate, and conversational quality—what metrics and evaluation procedures did you use?",
            "rationale": "To understand how the model's real-world performance and alignment were quantitatively and qualitatively evaluated."
        },
        {
            "followup_question": "What specific guardrail mechanisms did you implement within AWS Bedrock and Vertex AI (e.g., classifiers, filters, moderated prompts), and how do they handle false positives or blocked useful content?",
            "rationale": "To probe the safety implementation details and how trade-offs between safety and usability were managed."
        }
    ]
}

# Invalid response from OpenAI (list of dicts instead of JSON)
invalid_output_text = [
    {
        "followup_question": "Can you describe the RAG pipeline you implemented—retriever type, embedding model, vector store, and how you integrated retrieval results into prompts?",
        "rationale": "To assess the technical choices and how retrieval data was incorporated into prompt design for accuracy."
    },
    {
        "followup_question": "How did you measure and validate the chatbot's accuracy, hallucination rate, and conversational quality—what metrics and evaluation procedures did you use?",
        "rationale": "To understand how the model's real-world performance and alignment were quantitatively and qualitatively evaluated."
    },
    {
        "followup_question": "What specific guardrail mechanisms did you implement within AWS Bedrock and Vertex AI (e.g., classifiers, filters, moderated prompts), and how do they handle false positives or blocked useful content?",
        "rationale": "To probe the safety implementation details and how trade-offs between safety and usability were managed."
    }
]

# Test 1: Successful follow-up generation
def test_success():
    # Mock the OpenAI client to simulate a successful response
    with patch("api_backend.client.responses.create") as mock_create:
        mock_response = MagicMock()
        mock_response.status = "succeeded"
        mock_response.output_text = json.dumps(valid_output_text)
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
        # Check that output Content-Type header is application/json
        assert response.headers["content-type"] == "application/json"

# Test 2: Missing required field (question)
def test_missing_required_field():
    # Make a request missing the required 'question' field
    response = client.post("/interview/generate-followups", json=missing_question_request)
    # Check that FastAPI rejects request with 442 validation error
    assert response.status_code == 422
    # Check that output Content-Type header is application/json
    assert response.headers["content-type"] == "application/json"

# Test 3: Invalid field type (interview_type should be a list of strings)
def test_invalid_field_type():
    # Make a request with a wrong data type for 'interview_type'
    response = client.post("/interview/generate-followups", json=invalid_type_request)
    # Check that FastAPI rejects request with 442 validation error
    assert response.status_code == 422
    # Check that output Content-Type header is application/json
    assert response.headers["content-type"] == "application/json"

# Test 4: Valid request with missing optional fields
def test_missing_optional_fields():
    # Mock the OpenAI client to simulate a successful response
    with patch("api_backend.client.responses.create") as mock_create:
        mock_response = MagicMock()
        mock_response.status = "succeeded"
        mock_response.output_text = json.dumps(valid_output_text)
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
        # Check that output Content-Type header is application/json
        assert response.headers["content-type"] == "application/json"

# Test 5: GET method is not allowed
def test_get_method_not_allowed():
    # Make a request with the GET method
    response = client.get("/interview/generate-followups")
    # Check that FastAPI FastAPI rejects non-POST methods with 405
    assert response.status_code == 405
    # Check that output Content-Type header is application/json
    assert response.headers["content-type"] == "application/json"

# Test 6: Sending data with wrong Content-Type
def test_wrong_content_type():
    # Make a request as form-encoded instead of JSON
    response = client.post("/interview/generate-followups", content="question=Hi&answer=Hello", headers={"Content-Type": "application/x-www-form-urlencoded"})
    # Check that FastAPI rejects request with 442 validation error
    assert response.status_code == 422
    # Check that output Content-Type header is application/json
    assert response.headers["content-type"] == "application/json"

# Test 7: Empty json body
def test_empty_json():
    # Make a request with an empty json
    response = client.post("/interview/generate-followups", json={})
    # Check that FastAPI rejects request with 442 validation error
    assert response.status_code == 422
    # Check that output Content-Type header is application/json
    assert response.headers["content-type"] == "application/json"

# Test 8: Model returns incomplete status
def test_incomplete_status():
    # Mock the OpenAI client to simulate an incomplete response
    with patch("api_backend.client.responses.create") as mock_create:
        mock_response = MagicMock()
        mock_response.status = "incomplete"
        mock_response.incomplete_details.reason = "timeout"
        mock_create.return_value = mock_response
        # Make a request with a complete request
        response = client.post("/interview/generate-followups", json=complete_request)
        # Check for HTTP 500 error code
        assert response.status_code == 500
        # Check for correct error message
        assert response.json()["detail"]["message"] == "Model output incomplete."
        # Check that output Content-Type header is application/json
        assert response.headers["content-type"] == "application/json"

# Test 9: Model returns empty output
def test_empty_output():
    # Mock the OpenAI client to simulate an empty output
    with patch("api_backend.client.responses.create") as mock_create:
        mock_response = MagicMock()
        mock_response.status = "succeeded"
        mock_response.output_text = ""
        mock_create.return_value = mock_response
        # Make a request with a complete request
        response = client.post("/interview/generate-followups", json=complete_request)
        # Check for HTTP 500 error code
        assert response.status_code == 500
        # Check for correct error message
        assert response.json()["detail"]["message"] == "Model returned empty output."
        # Check that output Content-Type header is application/json
        assert response.headers["content-type"] == "application/json"

# Test 10: Model returns invalid JSON or unexpected structure (list of dicts instead of JSON)
def test_invalid_output():
    # Mock the OpenAI client to simulate an invalid output format
    with patch("api_backend.client.responses.create") as mock_create:
        mock_response = MagicMock()
        mock_response.status = "succeeded"
        mock_response.output_text = invalid_output_text
        mock_create.return_value = mock_response
        # Make a request with a complete request
        response = client.post("/interview/generate-followups", json=complete_request)
        # Check for HTTP 500 error code
        assert response.status_code == 500
        # Check for correct error message
        assert response.json()["detail"]["message"] == "Failed to parse output text." 
        # Check that output Content-Type header is application/json
        assert response.headers["content-type"] == "application/json"

# Test 11: Model returns empty follow-ups list
def test_empty_followups():
    # Mock the OpenAI client to simulate an empty follow-ups list
    with patch("api_backend.client.responses.create") as mock_create:
        mock_response = MagicMock()
        mock_response.status = "succeeded"
        mock_response.output_text = json.dumps({"followups": []})
        mock_create.return_value = mock_response
        # Make a request with a complete request
        response = client.post("/interview/generate-followups", json=complete_request)
        # Check for HTTP 500 error code
        assert response.status_code == 500
        # Check for correct error message
        assert response.json()["detail"]["message"] == "Model returned empty follow-ups list." 
        # Check that output Content-Type header is application/json
        assert response.headers["content-type"] == "application/json"

# Test 12: OpenAI exception handling
def test_openai_exception():
    # Mock the OpenAI client to simulate an exception being thrown
    with patch("api_backend.client.responses.create", side_effect=Exception("API down")):
        # Make a request with a complete request
        response = client.post("/interview/generate-followups", json=complete_request)
        # Check for HTTP 500 error code
        assert response.status_code == 500
        # Check for correct error message
        assert response.json()["detail"]["message"] == "OpenAI client failed." 
        # Check that output Content-Type header is application/json
        assert response.headers["content-type"] == "application/json"