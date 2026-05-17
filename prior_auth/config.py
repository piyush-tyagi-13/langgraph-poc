import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def create_llm() -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
    )
