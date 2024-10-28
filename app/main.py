# app/main.py

from fastapi import FastAPI
from app.routes import chat  # Importing chat routes

app = FastAPI()

# Include the chat routes
app.include_router(chat.router)

# Root endpoint for testing
@app.get("/")
async def root():
    return {"message": "Haystack RAG with FastAPI is running"}