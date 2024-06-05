from .chatbot import Chatbot
from openai import AsyncOpenAI
from pydantic import BaseModel

llm = "gpt-4o"

chatbot = Chatbot()

client = AsyncOpenAI()

class CreateAssistant(BaseModel):
    name: str
    instruction: str

class CreateMessage(BaseModel):
    assistant_id: str
    content: str

functions = [
    {
        "type": "function",
        "function": {
            "name": "answer_query",
            "description": "Answer customer query",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The customer's query question"},
                },
                "required": ["question"]
            }
        }
    }
]

thread_storage = {}

