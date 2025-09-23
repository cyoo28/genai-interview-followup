import pytest
from openai import OpenAI
from api_backend import call_openai, FollowUpResponse
import numpy as np
import json

# Initialize OpenAI client (uses credentials configured in environment)
client = OpenAI()

# Utility function to compute cosine similarity between two vectors
def cosine_sim(a, b):
    # Convert both vectors into numpy arrays
    a, b = np.array(a), np.array(b)
    # Calculate the cosine similarity
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Utility function to get embedding vector using OpenAI
def get_embedding(text, model="text-embedding-3-small"):
    # Retrieve embedding vector for given text from OpenAI
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding

def test_complete():
    # Test purpose
    purpose = "Testing the OpenAI API call in api_backend.py with a complete request"

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

    # Split the candidate's answer by sentence
    answer_split = [s.strip() for s in payload["answer"].split(".") if s.strip()]
    # Create embedding for each sentence
    answer_embs = [get_embedding(s) for s in answer_split]

    # Prepare results with header (test purpose and request payload)
    results = []
    header = (
        f"Purpose: {purpose}\n"
        f"Payload:\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n"
    )
    results.append(header)
    # Compute the maximum cosine similarity for each followup
    for followup in followups.followups:
        # Create embedding for followup
        followup_emb = get_embedding(followup.followup_question)
        # Compute cosine similarity between followup embedding and each answer sentence embedding
        sims = [cosine_sim(np.array(followup_emb), np.array(ae)) for ae in answer_embs]
        # Find the maximum cosine similarity
        max_sim = max(sims)
        # Determine if followup is similar enough
        status = "PASSED" if max_sim >= 0.4 else "FAILED"
        # Append result
        line = f"Follow-up: {followup.followup_question}\nRationale: {followup.rationale}\nMax Cosine Sim: {max_sim:.3f}\nStatus: {status}\n"
        results.append(line)
    # Write results to a file
    with open("test_complete_results.txt", "w", encoding="utf-8") as f:
        f.writelines([r + "\n" for r in results])