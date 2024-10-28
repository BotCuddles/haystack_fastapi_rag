# app/config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    DOCUMENT_STORE_DIR = os.getenv("DOCUMENT_STORE_DIR", "document_store")
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")

settings = Settings()
