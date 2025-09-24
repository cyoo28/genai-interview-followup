# Genai Followup Question Backend
This project creates a FastAPI backend that generates follow up questions for an interviewer based on the original question and the candidate's answer. This project was completed in September 2025.

## Table of Contents
* [Problem Statement](#problem-statement)
* [Solution Methodology](#solution-methodology)
* [Testing](#testing)
  * [FastAPI Tests](#fastapi-tests)
  * [OpenAI Tests](#openai-tests)
* [Further Considerations](#further-considerations)

## Problem Statement
This project needs to implement a FastAPI backend that generates follow-up questions for interviews. The service must expose a POST endpoint that accepts an interviewer’s original question, a candidate’s response, and optional context (e.g. the role the candidate is applying for and the type of interview).

The backend should:
1. Validate incoming requests to ensure required fields are present and formatted.
2. Call OpenAI API using a system prompt to generate at least one follow-up question.
3. Return a structured JSON response that includes both the generated follow-up question(s) and rationale.

The API will enable interviewers to generate follow-up questions, improving the flow of interviews.

## Solution Methodology
The solution was designed to generate relevant follow-up interview questions from candidate answers in a structured and reliable way. The methodology includes:
- **Framework and API Design**:  
  Built with **FastAPI** to provide a lightweight endpoint (`/interview/generate-followups`). Input validation and schema enforcement are handled using **Pydantic** models.
- **Model Integration**:  
  The backend calls the **OpenAI Responses API** (`gpt-5-mini`) with carefully designed system instructions that constrain outputs to 1–3 concise follow-up questions with rationales. Instructions enforce safe, professional, and JSON-formatted outputs.
- **Validation and Error Handling**:  
  Model outputs are parsed and validated against a strict Pydantic schema (`FollowUpResponse`). Common error cases are explicitly handled (incomplete responses, empty outputs, invalid JSON, or missing follow-ups), with descriptive `500 Internal Server Error` responses returned for debugging.
- **Testing and Quality Assurance**:  
  Unit and integration tests were written using **pytest**. Tests cover request validation, error handling, and model behavior across valid, invalid, and edge-case inputs.
- **Deployment and Execution**:  
  The API can be served locally with **Uvicorn**, making it easy to containerize or deploy in a production environment. Before running, install the required dependencies for reproducibility:
  - fastapi==0.117.1
  - openai==1.108.1
  - pydantic==2.11.9
  - pytest==8.4.2
  - uvicorn==0.36.0
  
  Example command to run locally:
  ```bash
  uvicorn api_backend:app --reload

## Testing
This project includes tests to validate the FastAPI backend and the OpenAI API integration. These tests ensure that the backend behaves as expected for various inputs.

### FastAPI Tests
These tests validate the backend endpoint to ensure correct request/response handling and error cases:
- **Valid requests**:
  - Complete payload                               → `200 OK` with JSON containing follow-up questions
  - Minimal payload (exclude optional fields)      → `200 OK` with JSON containing follow-up questions
- **Invalid requests**:  
  - Missing required field (question)              → `422 Unprocessable Entity` with validation error  
  - Invalid field type (interview_type not list)   → `422 Unprocessable Entity` with field/type mismatch  
  - Empty JSON body                                → `422 Unprocessable Entity` with "field required" messages  
  - Wrong Content-Type (form-encoded)              → `422 Unprocessable Entity` with parsing error  
- **Method handling**:
  - GET requests                                   → `405 Method Not Allowed` with error message indicating method not supported  
- **Model error handling**:  
  - Incomplete status                              → `500 Internal Server Error` with `"error": "Model did not complete successfully"`  
  - Empty output                                   → `500 Internal Server Error` with `"error": "No output returned from model"`  
  - Invalid output format (non-JSON)               → `500 Internal Server Error` with `"error": "Invalid response format from model"`  
  - Empty follow-up list                           → `500 Internal Server Error` with `"error": "No follow-up questions generated"`  
  - OpenAI client exception                        → `500 Internal Server Error` with exception message wrapped in `"error"`  

### OpenAI Tests
These tests validate the actual model integration via `call_openai`:
- **Complete request**: Full payload with role & interview type
- **Minimal request**: Payload with only required fields
- **Vague answer**: Very short/unclear candidate response 
- **Irrelevant answer**: Off-topic response
For each follow-up, cosine similarity is computed against the candidate answer sentences to verify semantic relevance (`PASSED` if ≥0.4).
For each test, results are written (payload, follow-ups, cosine similarity scores, pass/fail) to a text file.  

## Further Considerations
There are several areas where this project could be extended or improved:

- **Cosine similarity challenges**: Current relevance checks rely on cosine similarity between embeddings of candidate answers and follow-up questions. It can be difficult to evaluate similarity when a candidate's answer is long or covers multiple subjects. Improvements could include using sentence transformers fine-tuned for semantic relatedness.
  - **Promptfoo integration**: Adding [promptfoo](https://www.promptfoo.dev/docs/intro/) could allow more systematic evaluation of prompts and outputs across a test suite of inputs.
- **End-to-end testing**: Current tests cover the API and model integration separately. Conducting extensive testing for the entire code together could ensure that the backend works as expected.
- **Observability/monitoring**: Adding logging, request/response tracing, and metrics could help diagnose issues and improve reliability.
- **Security**: Input validation is currently handled by Pydantic. Additional safeguards like request rate limiting and input length checks could improve security.
- **Model improvements**: Exploring larger models may improve quality of follow-up questions. RAG (retrieval-augmented generation) could also be considered for domain-specific interviewing contexts.
