# utils/config.py
from dotenv import load_dotenv
import os

load_dotenv()

def get_api_key(name):
    key = os.getenv(name)
    if not key:
        raise ValueError(f"{name} is not set in .env file.")
    return key
