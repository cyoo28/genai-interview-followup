# Genai Followup Question Backend
This project creates a FastAPI backend that generates follow up questions for an interviewer based on the original question and the candidate's answer. This project was completed in September 2025.

## Table of Contents
* [Problem Statement](#problem-statement)
* [Solution Methodology](#solution-methodology)
* [Solution Methodology](#testing)
* [Testing](#other-considerations)

## Problem Statement
The goal of this project is to implement a FastAPI backend that generates relevant follow-up questions for interviews. The service must expose a POST endpoint that accepts an interviewer’s original question, a candidate’s response, and optional context (e.g. the role the candidate is applying for and the type of interview). The backend should:

1. Validate incoming requests to ensure required fields are present and properly formatted.
2. Call OpenAI API with a clear system prompt to generate at least one follow-up questions.
3. Return a structured JSON response that includes both the generated follow-up question(s) and rationale.

The final API enables interviewers to generate relevant follow-up questions, improving the flow of interview conversations.

## Solution Methodology

## Testing

## Other Considerations
