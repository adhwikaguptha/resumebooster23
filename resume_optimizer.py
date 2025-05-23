# ai_processor/resume_optimizer.py

import json
from ai_processor.ai_router import query_ai_model
from document_processor import calculate_ats_score
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access keys
nomic_api_key = os.getenv("nk-F7G7L6HC3Us-yTrZn2lFMUP31qka0fl_ATcNhXWKf-g")
openrouter_api_key = os.getenv("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")

def generate_resume_feedback(resume_text, job_description, provider="groq", model="llama3-70b-8192"):
    """
    Process resume + JD through selected LLM API and return feedback + ATS score.

    Returns:
        dict: {
            "suggestions": [str, ...],
            "optimized_resume": str,
            "ats_score": float
        }
    """
    prompt = f"""
You are an expert in resume screening and improvement.

Given the following job description:
----
{job_description}
----

And the following resume:
----
{resume_text}
----

Provide a bullet-point list of suggestions to improve the resume specifically for this role.
Be specific. Include missing skills, formatting, or content improvements.

Then rewrite the resume tailored to the job.

Output valid JSON like:
{{
  "suggestions": ["...", "..."],
  "optimized_resume": "..."
}}
    """

    try:
        # Call AI model
        raw_response = query_ai_model(prompt, provider=provider, model=model)
        print("DEBUG raw_response:", raw_response)

        # Try to parse JSON response from the model output
        try:
            response = json.loads(raw_response)
        except json.JSONDecodeError:
            print("⚠️ Failed to parse JSON from AI response. Returning raw text.")
            response = {
                "suggestions": ["⚠️ AI returned unstructured text. Please try again."],
                "optimized_resume": raw_response
            }

    except Exception as e:
        print(f"❌ Error during AI model call: {e}")
        response = {
            "suggestions": [f"An error occurred while querying the AI: {e}"],
            "optimized_resume": ""
        }

    # Compute ATS Score
    ats_score = calculate_ats_score(resume_text, job_description)

    result = {
        "suggestions": response.get("suggestions", []),
        "optimized_resume": response.get("optimized_resume", ""),
        "ats_score": ats_score
    }

    return result
