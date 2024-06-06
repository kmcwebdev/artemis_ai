## Project Structure Reference

For detailed information on structuring your FastAPI project, please refer to the official documentation:

[FastAPI Project Structure Guide](https://fastapi.tiangolo.com/tutorial/bigger-applications/)

***

# Project File Structure
```
├── app
│   ├── __init__.py       
│   ├── main.py           
│   ├── chatbot.py        
│   ├── dependencies.py   
│   └── routers
│   │   ├── __init__.py   
│   │   └── users.py      
│   └── internal
│       ├── __init__.py   
│       └── admin.py      
```

# app
#### __init__.py
> Empty placeholder to indicate app directory a python package
#### main.py
> Initialize FastAPI application
```python
 from fastapi import FastAPI
 from fastapi.middleware.cors import CORSMiddleware
 from app.routers import users

 app = FastAPI()

 app.add_middleware(
     CORSMiddleware,
     allow_origins=["http://localhost:3000"],
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],
 )

 app.include_router(users.router)

 @app.get("/")
 async def root():
     return {"Hello": "World"}
```  
#### chatbot.py
> Chatbot functionality
```python
> Chatbot functionality
from dotenv import load_dotenv
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

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
```
#### dependencies.py
> Defines dependencies used across the application  
```python
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
```
***
# routers
#### __init__.py
> Empty placeholder to indicate routers a Python subpackage
#### users.py
> Handle API routes for assistants, threads, and messages  
```python
import json
from fastapi import APIRouter, HTTPException
from app.dependencies import functions, chatbot, llm, thread_storage, client
from app.dependencies import CreateAssistant
from app.dependencies import CreateMessage

router = APIRouter()

@router.get("/api/assistant/{assistant_id}")
async def get_assistant_info(assistant_id: str):
    assistant = await client.beta.assistants.retrieve(assistant_id=assistant_id)

    return {
        "assistant_id": assistant.id,
        "name": assistant.name,
        "instruction": assistant.instructions
    }

@router.post("/api/assistant")
async def create_new_assistant(data: CreateAssistant):
    assistant = await client.beta.assistants.create(
        name=data.name,
        instructions=data.instruction,
        tools=functions,
        temperature=0,
        model=llm,
    )

    return {
        "assistant_id": assistant.id
    }

@router.post("/api/threads/{assistant_id}")
async def create_thread(assistant_id: str):
    thread = await client.beta.threads.create()

    await client.beta.threads.messages.create(
        thread_id=thread.id,
        content="Greet the user and ask how you can help.",
        role="user",
        metadata={
            "type": "hidden"
        }
    )

    run = await client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id
    )

    # Store the thread ID associated with the assistant ID
    if assistant_id not in thread_storage:
        thread_storage[assistant_id] = []
    thread_storage[assistant_id].append(thread.id)

    return {
        "thread_id": thread.id,
        "run_id": run.id,
        "status": run.status
    }

@router.post("/api/threads/{thread_id}/messages")
async def create_message(thread_id: str, message: CreateMessage):
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

@router.get("/api/threads/{thread_id}/runs/{run_id}/status")
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
        }

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                user_input=function_args.get("question")
            )

            tools_output.append({
                "tool_call_id": tool_call.id,
                "output": function_response
            })

        run = await client.beta.threads.runs.submit_tool_outputs(
            run_id=run_id,
            thread_id=thread_id,
            tool_outputs=tools_output
        )

    return {
        "thread_id": thread_id,
        "run_id": run.id,
        "status": run.status
    }

@router.get("/api/threads/{thread_id}/messages")
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

@router.get("/api/assistant/{assistant_id}/threads")
async def get_threads_for_assistant(assistant_id: str):
    if assistant_id not in thread_storage:
        raise HTTPException(status_code=404, detail="Assistant ID not found")
    
    return {"assistant_id": assistant_id, "thread_ids": thread_storage[assistant_id]}
```
***
# internal
#### __init__.py
> Empty placeholder to indicate internal a Python subpackage
#### admin.py 
> Handles internal administrative tasks, storage management, and chat history

***

## main

```python
app = FastAPI()
```
* Initializes a FastAPI application named app

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
* Adding CORS (Cross-Origin Resource Sharing) middleware to the application
* Specifies the allowed origins for CORS 
* Allows credentials (such as cookies) to be included in requests
* Specify the allowed HTTP methods and headers  

```python
app.include_router(users.router)
```
* Includes routes from the "users" module

```python
@app.get("/")
async def root():
    return {"Hello": "World"}
```
* Defines a GET endpoint at the root URL ("/") and returns a JSON response 

***

## chatbot

```python
class Chatbot:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
        self.rag_chain = self.setup_rag_chain()
```
* Initialize the Chatbot class
* Initializes a LLM
* Sets up a retrieval-augmented generation (RAG) chain

```python
def setup_rag_chain(self):
    loader = PyPDFLoader("Technology Services FAQs.pdf")
    pages = loader.load_and_split()
```
* Loads and splits the PDF file into individual pages

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(pages)
```
* Split documents into chunks of 1000 characters with an overlap of 200 characters
* Splits the loaded pages into smaller text chunks

```python
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
```
* Creates a vector store from the document splits using OpenAI embeddings

```python
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
```
* Sets up the vector store as a retriever
* Pulls a prompt template from a hub

```python
def format_docs(docs):
       return "\n\n".join(doc.page_content for doc in docs)
```
* Format documents into a single string

```python
 rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | self.llm
        | StrOutputParser()
    )
    return rag_chain
```
* Creates a RAG chain that processes the user input
* Uses the retriever and format_docs function to format the context
* Passes the question through and applies the prompt
* Parses the output into a string

```python
def get_response(self, user_input):
    try:
        response = self.rag_chain.invoke(user_input)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```
* Invokes the RAG chain with the user input to generate a response
* Catches any exceptions and raises an HTTPException with a status code of 500, including the error details

***

## dependencies

```python
chatbot = Chatbot()
```
* Creates an instance of the Chatbot class

```python
client = AsyncOpenAI()
```
* Creates an instance of the AsyncOpenAI client for asynchronous interactions with OpenAI

```python
class CreateAssistant(BaseModel):
    name: str
    instruction: str
```
* Defines a Pydantic model with name and instruction fields

```python
class CreateMessage(BaseModel):
    assistant_id: str
    content: str
```
* Defines a Pydantic model with assistant_id and content fields

```python
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
```
* Defines a list of functions, where each function includes:
* Function name
* Description of the function's purpose
* Required parameters for the function

```python
thread_storage = {}
```
* Initializes an empty dictionary to store thread IDs associated with assistant IDs

***

## users

```python
@router.get("/api/assistant/{assistant_id}")
async def get_assistant_info(assistant_id: str):
    assistant = await client.beta.assistants.retrieve(assistant_id=assistant_id)

    return {
        "assistant_id": assistant.id,
        "name": assistant.name,
        "instruction": assistant.instructions
    }
```
* Retrieves information about a specific assistant based on the assistant_id

```python
@router.post("/api/assistant")
async def create_new_assistant(data: CreateAssistant):
    assistant = await client.beta.assistants.create(
        name=data.name,
        instructions=data.instruction,
        tools=functions,
        temperature=0,
        model=llm,
    )

    return {
        "assistant_id": assistant.id
    }
```
* Creates a new assistant with specified details and returns the new assistant_id

```python
@router.post("/api/threads/{assistant_id}")
async def create_thread(assistant_id: str):
    thread = await client.beta.threads.create()

    await client.beta.threads.messages.create(
        thread_id=thread.id,
        content="Greet the user and ask how you can help.",
        role="user",
        metadata={
            "type": "hidden"
        }
    )

    run = await client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant_id
    )

    # Store the thread ID associated with the assistant ID
    if assistant_id not in thread_storage:
        thread_storage[assistant_id] = []
    thread_storage[assistant_id].append(thread.id)

    return {
        "thread_id": thread.id,
        "run_id": run.id,
        "status": run.status
    }
```
* Creates a new thread_id for the specified assistant

```python
@router.post("/api/threads/{thread_id}/messages")
async def create_message(thread_id: str, message: CreateMessage):
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
```
* Adds a new message to an existing thread_id

```python
@router.get("/api/threads/{thread_id}/runs/{run_id}/status")
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
        }

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                user_input=function_args.get("question")
            )

            tools_output.append({
                "tool_call_id": tool_call.id,
                "output": function_response
            })

        run = await client.beta.threads.runs.submit_tool_outputs(
            run_id=run_id,
            thread_id=thread_id,
            tool_outputs=tools_output
        )

    return {
        "thread_id": thread_id,
        "run_id": run.id,
        "status": run.status
    }
```
* Retrieves the status of a specific run within a thread_id

```python
@router.get("/api/threads/{thread_id}/messages")
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
```
* Retrieves messages from a specific thread_id, displaying the conversation history between the user and the assistant.

```python
@router.get("/api/assistant/{assistant_id}/threads")
async def get_threads_for_assistant(assistant_id: str):
    if assistant_id not in thread_storage:
        raise HTTPException(status_code=404, detail="Assistant ID not found")
    
    return {"assistant_id": assistant_id, "thread_ids": thread_storage[assistant_id]}
```
* Retrieves the list of threads_ids associated with a specific assistant based of assistant_id
