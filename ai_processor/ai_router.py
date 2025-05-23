# ai_processor/ai_router.py
import os
import requests
from utils.config import get_api_key

def call_groq(prompt, system_prompt="", model="llama3-70b-8192"):
    return _call_openai_style(
        url="https://api.groq.com/openai/v1/chat/completions",
        key_env="GROQ_API_KEY",
        model=model,
        prompt=prompt,
        system_prompt=system_prompt
    )

def call_together(prompt, system_prompt="", model="togethercomputer/Command-R+"):
    return _call_openai_style(
        url="https://api.together.xyz/v1/chat/completions",
        key_env="TOGETHER_API_KEY",
        model=model,
        prompt=prompt,
        system_prompt=system_prompt
    )

def call_openrouter(prompt, system_prompt="", model="openai/gpt-4-turbo"):
    return _call_openai_style(
        url="https://openrouter.ai/api/v1/chat/completions",
        key_env="OPENROUTER_API_KEY",
        model=model,
        prompt=prompt,
        system_prompt=system_prompt
    )

def call_huggingface(prompt, system_prompt="", model="mistralai/Mistral-7B-Instruct-v0.1"):
    """
    Call Hugging Face hosted inference endpoint using Chat-like format.
    Assumes use of HuggingFace Inference API.
    """
    api_key = get_api_key("HUGGINGFACE_API_KEY")
    url = f"https://api-inference.huggingface.co/models/{model}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": f"[INST] <<SYS>> {system_prompt} <</SYS>> {prompt} [/INST]",
        "parameters": {
            "return_full_text": False,
            "temperature": 0.3,
            "max_new_tokens": 1024
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()

    # Extract response text
    if isinstance(result, list):
        return result[0]["generated_text"].strip()
    else:
        raise ValueError(f"Hugging Face API Error: {result}")

def _call_openai_style(url, key_env, model, prompt, system_prompt):
    api_key = get_api_key(key_env)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1500,
        "temperature": 0.3
    }
    response = requests.post(url, headers=headers, json=body)
    result = response.json()
    return result['choices'][0]['message']['content'].strip()
# ai_processor/ai_router.py

def query_ai_model(prompt, provider="groq", model="mixtral"):
    if provider == "groq":
        return parse_json_response(call_groq(prompt, model=model))
    elif provider == "together":
        return parse_json_response(call_together(prompt, model=model))
    elif provider == "openrouter":
        return parse_json_response(call_openrouter(prompt, model=model))
    elif provider == "huggingface":
        return parse_json_response(call_huggingface(prompt, model=model))
    else:
        raise ValueError(f"Unsupported provider: {provider}")
import json
import re

def parse_json_response(response_text):
    """
    Try to extract and parse JSON from an LLM response.
    This handles cases where the response has extra commentary or formatting.
    """
    try:
        # Attempt direct parsing
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON using regex (best-effort fallback)
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Fallback structure if parsing fails
        return {
            "suggestions": ["Could not parse response as JSON."],
            "optimized_resume": response_text
        }

