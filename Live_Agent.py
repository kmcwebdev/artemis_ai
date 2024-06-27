from dotenv import load_dotenv
from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
from fastapi.middleware.cors import CORSMiddleware
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = "gpt-4o"

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
    },
    {
        "type": "function",
        "function": {
            "name": "connect_to_live_agent",
            "description": "Connect the user to a live agent",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_message": {"type": "string", "description": "The user's message"}
                },
                "required": ["user_message"]
            }
        }
    }
]

# Initialize the AsyncOpenAI client
client = AsyncOpenAI()

class CreateMessage(BaseModel):
    assistant_id: str
    content: str

class Chatbot:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
        self.rag_chain = self.setup_rag_chain()

    def setup_rag_chain(self):
        loader = PyPDFLoader("Technology Services FAQs.pdf")
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(pages)

        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

        retriever = vectorstore.as_retriever()
        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain

    def get_response(self, user_input):
        try:
            response = self.rag_chain.invoke(user_input)
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

chatbot = Chatbot()


def connect_to_live_agent(user_message):
    """Connect the user to a live agent"""
    return json.dumps({"message": "Would you like to connect to a live agent to assist you further?"})


"""Use the language model to determine if the user is confused""" 
async def is_user_confused(user_message):
    messages = [
        {"role": "system", "content": "Determine from the following message if user is confused and requires further assistance."},
        {"role": "user", "content": user_message}
    ]   
    response = await client.chat.completions.create(
        model=llm,
        messages=messages,
        temperature = 0.5
    )


@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/api/assistant")
async def create_new_assistant():
    assistant = await client.beta.assistants.create(
        name="Customer FAQ Assistant",
        instructions="You are an assistant specialized in answering customer queries",
        tools=functions,
        temperature=0.1,
        model=llm,
    )

    return {
        "Assistant_id": assistant.id
    }

@app.post("/api/threads/{assistant_id}")
async def create_thread(assistant_id: str):
    thread = await client.beta.threads.create()

    await client.beta.threads.messages.create(
        thread_id=thread.id,
        content="Greet the user and tell it about yourself and ask it what it is looking for.",
        role="user",
        metadata={
            "type": "hidden"
        }
    )

    run = await client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id
    )

    return {
        "thread_id": thread.id,
        "run_id": run.id,
        "status": run.status
    }

@app.post("/api/threads/{thread_id}/messages")
async def create_message(thread_id: str, message: CreateMessage):
    if await is_user_confused(message.content):
        run = await client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=message.assistant_id,
            message={"content": "connect_to_live_agent", "user_message": message.content}
        )
    else:
        # If user is not confused, create a regular message
        await client.beta.threads.messages.create(
            thread_id=thread_id,
            content=message.content,
            role="user"
        )
        
        run = await client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=message.assistant_id
        )

    return {
        "run_id": run.id,
        "thread_id": thread_id,
        "status": run.status,
        "last_error": run.last_error
    }

@app.get("/api/threads/{thread_id}/runs/{run_id}/status")
async def get_status(thread_id: str, run_id: str):
    run = await client.beta.threads.runs.retrieve(
        thread_id=thread_id, 
        run_id=run_id
    )

    if (run.status == "requires_action"):
        print("Function needs to be invoked")
        tool_calls = run.required_action.submit_tool_outputs.tool_calls

        tools_output = []

        available_functions = {
            "answer_query": chatbot.get_response,
            "connect_to_live_agent": connect_to_live_agent
        }

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            if function_name == "answer_query":
                function_response = function_to_call(
                    user_input=function_args.get("question")
                    )
            elif function_name == "connect_to_live_agent":
                function_response = function_to_call(
                    function_args.get("user_message")
                )

            tools_output.append({
                "tool_call_id": tool_call.id,
                "output": json.dumps(function_response)
            })

        run = await client.beta.threads.runs.submit_tool_outputs(
            run_id=run_id,
            thread_id=thread_id,
            tool_outputs=tools_output
        )

    return {
        "thread_id": thread_id,
        "run_id": run_id,
        "status": run.status
    }

@app.get("/api/threads/{thread_id}/messages")
async def get_thread_messages(thread_id: str):
    messages = await client.beta.threads.messages.list(
        thread_id=thread_id
    )

    result = [
        {
            "content": message.content[0].text.value,
            "role": message.role,
            "hidden": "type" in message.metadata and message.metadata["type"] == "hidden",
            "id": message.id,
            "created_at": message.created_at,
        }
        for message in messages.data
    ]

    return result