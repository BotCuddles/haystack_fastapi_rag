# app/routes/chat.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.pipelines import create_conversational_rag_pipeline, download_and_process_documents
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack.dataclasses import ChatMessage
from typing import Dict

router = APIRouter()
session_memory_stores: Dict[str, InMemoryChatMessageStore] = {}

# Download and process documents once at startup
download_and_process_documents()

# Request model for user input
class QueryInput(BaseModel):
    session_id: str
    question: str

# Initialize session endpoint
@router.post("/init_session/")
async def init_session():
    session_id = str(len(session_memory_stores) + 1)
    session_memory_stores[session_id] = InMemoryChatMessageStore()
    return {"session_id": session_id}

# Chat query endpoint
@router.post("/ask/")
async def ask_question(input: QueryInput):
    session_id = input.session_id
    question = input.question

    if session_id not in session_memory_stores:
        raise HTTPException(status_code=404, detail="Session not found")

    memory_store = session_memory_stores[session_id]
    conversational_rag = create_conversational_rag_pipeline(memory_store)

    # Define system and user messages
    system_message = ChatMessage.from_system("You are a helpful AI assistant using provided supporting documents and conversation history to assist humans")
    user_message = ChatMessage.from_user(question)
    messages = [system_message, user_message]

    # Run the pipeline for each query
    result = conversational_rag.run(data={
        "query_rephrase_prompt_builder": {"query": question},
        "prompt_builder": {"template": messages, "query": question},
        "memory_joiner": {"value": [user_message]}
    }, include_outputs_from=["llm", "query_rephrase_llm"])

    search_query = result['query_rephrase_llm']['replies'][0]
    assistant_response = result['llm']['replies'][0].content

    return {"search_query": search_query, "answer": assistant_response}
