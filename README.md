# Project Structure Reference

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
```

***
# routers
#### __init__.py
> Empty placeholder to indicate routers a Python subpackage

#### users.py
> Handle API routes for assistants, sessions, and messages  
```python
import json
from fastapi import APIRouter
from app.dependencies import functions, chatbot, llm, client
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
        content="Greet the user and tell it about yourself and ask how you can help.",
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
        tools_output = []
        tool_calls = run.required_action.submit_tool_outputs.tool_calls

        available_functions = {
            "answer_query": chatbot.get_response,
        }

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "answer_query":
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
```

***
# internal
#### __init__.py
> Empty placeholder to indicate internal a Python subpackage

#### admin.py 
> Handles internal administration, storage management, and chat history

***
# Code Breakdown Overview
# main

> ### Imports
```python
 from fastapi import FastAPI
 from fastapi.middleware.cors import CORSMiddleware
 from app.routers import users
```
Import necessary dependencies and modules

***
> ### Initialize FastAPI and Router
```python
app = FastAPI()

app.include_router(users.router)
```
Initialize a FastAPI application named app
Include routes from the "users" module

***
> ### CORS Middleware configuration
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
Add CORS (Cross-Origin Resource Sharing) middleware to the application

Specify the allowed origins for CORS 

Allow credentials to be included in requests

Specify the allowed HTTP methods and headers  

Include routes from the "users" module

***
> ### Root Endpoint
```python
@app.get("/")
async def root():
    return {"Hello": "World"}
```
Define a GET endpoint at the root URL and returns a JSON response 

***
# chatbot

> ### Imports
```python
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
```
Import necessary dependencies and modules and load environment variables

***
> ### Class
```python
class Chatbot:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
        self.rag_chain = self.setup_rag_chain()
```
Initialize the Chatbot class

Initialize an LLM

Set up a retrieval-augmented generation (RAG) chain

***
> ### Load File
```python
def setup_rag_chain(self):
    loader = PyPDFLoader("Technology Services FAQs.pdf")
    pages = loader.load_and_split()
```
Load and split the PDF file into individual pages

***
> ### Splitting
```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(pages)
```
Split documents into chunks of 1000 characters with an overlap of 200 characters

Split the loaded pages into smaller text chunks

***
> ### Storing
```python
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
```
Create a vector store from the document splits using OpenAI embeddings

***
> ### Retreival 
```python
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
```
Set up the vector store as a retriever

Pull a prompt template from a hub

***
> ### Formatting
```python
def format_docs(docs):
       return "\n\n".join(doc.page_content for doc in docs)
```
Format documents into a single string

***
> ### RAG Chain
```python
 rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | self.llm
        | StrOutputParser()
    )
    return rag_chain
```
Create a RAG chain that processes the user input

Use the retriever and format_docs function to format the context

Pass the question through and applies the prompt

Parse the output into a string

***
> ### Response
```python
def get_response(self, user_input):
    try:
        response = self.rag_chain.invoke(user_input)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```
Invoke the RAG chain with the user input to generate a response

Catch any exceptions and raises an HTTPException with a status code of 500, including the error details

***
# dependencies
> ### Imports
```python
from .chatbot import Chatbot
from openai import AsyncOpenAI
from pydantic import BaseModel
```
Import necessary dependencies
Import Chatbot class from chatbot module

***
> ### Chatbot and OpenAI Initialization
```python
llm = "gpt-4o"

chatbot = Chatbot()

client = AsyncOpenAI()
```
Specify LLM model

Create an instance of the Chatbot class

Create an instance of the AsyncOpenAI client for asynchronous interactions with OpenAI

***
> ### Assistant & Message Model Definitions
```python
class CreateAssistant(BaseModel):
    name: str
    instruction: str

class CreateMessage(BaseModel):
    assistant_id: str
    content: str
```
Define a Pydantic model "CreateAssistant" with name and instruction fields

Define a Pydantic model "CreateMessage" with assistant_id and content fields

***
> ### Assistant Function Definitions
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
Define a list of functions, where each function includes:
* Function name   
* Description of the function's purpose
* Required parameters for the function

***
# users
> ### Imports
```python
import json
from fastapi import APIRouter
from app.dependencies import functions, chatbot, llm, client
from app.dependencies import CreateAssistant
from app.dependencies import CreateMessage

router = APIRouter()
```
Import necessary modules and dependencies

Create instance of APIRouter

***
> ### Get assistant info
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
GET request at the endpoint /api/assistant/{assistant_id}.

Define an asynchronous function named get_assistant_info that takes an assistant_id as a path parameter.

Retrieve an assistant object from a client's beta assistants using the provided assistant_id.

Return a JSON response containing the assistant's ID, name, and instructions.

***
> ### Create new assistant
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
A POST request to the /api/assistant endpoint.

Define an asynchronous function named create_new_assistant that takes an argument data of type CreateAssistant.

Creates a new assistant using the client.beta.assistants.create method, passing various parameters:    

After the assistant is created, the function returns a dictionary containing the assistant_id, which is the ID of the newly created assistant.

***
> ### Create new thread
```python
@router.post("/api/threads/{assistant_id}")
async def create_thread(assistant_id: str):
    thread = await client.beta.threads.create()

    await client.beta.threads.messages.create(
        thread_id=thread.id,
        content="Greet the user and tell it about yourself and ask how you can help.",
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
```
POST request to "/api/threads/{assistant_id}".

Create a new thread using the client.beta.threads.create() method.

Create a new message in the thread with the content. This message is hidden from the user as indicated by the metadata.

Initiate a run of the assistant in the thread using the client.beta.threads.runs.create_and_poll() method.

Return a JSON response containing the thread ID, run ID, and the status of the run.

***
> ### Create new message
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
POST requests at the endpoint /api/threads/{thread_id}/messages.

Define an asynchronous function create_message that takes two parameters: thread_id (a string) and message (an instance of CreateMessage).

Create a new message in a specific thread using the client.beta.threads.messages.create method. The message content and role are taken from the message parameter.

Initiate a run of the assistant in the specified thread using the client.beta.threads.runs.create_and_poll method. The assistant ID is taken from the message parameter.

Return a dictionary containing the run ID, thread ID, run status, and the last error (if any) from the run.

***
> ### Get status of run
```python
@router.get("/api/threads/{thread_id}/runs/{run_id}/status")
async def get_status(thread_id: str, run_id: str):
    run = await client.beta.threads.runs.retrieve(
        thread_id=thread_id, 
        run_id=run_id
    )

    if (run.status == "requires_action"):
        tools_output = []
        tool_calls = run.required_action.submit_tool_outputs.tool_calls

        available_functions = {
            "answer_query": chatbot.get_response,
        }

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "answer_query":
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
GET request at the endpoint /api/threads/{thread_id}/runs/{run_id}/status.

Retrieve the status of a specific run in a thread using the client.beta.threads.runs.retrieve method.

If the run status is "requires_action", it processes the required actions:
* initialize an empty list tools_output and retrieves the tool calls from the run.
* Define a dictionary available_functions mapping function names to their implementations.
* Iterate over each tool call, retrieves the function name and arguments, and calls the corresponding function.
* If the function name is "answer_query", it calls the function with the question from the arguments.
* Append the tool call ID and function response to tools_output.
* Submit the tool outputs using the client.beta.threads.runs.submit_tool_outputs method.

Return a dictionary containing the thread ID, run ID, and run status.

***
> ### Get all messages in thread
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
GET request at the endpoint /api/threads/{thread_id}/messages.

Asynchronous function named get_thread_messages that takes a thread_id as a parameter.

Retrieve a list of messages from a specific thread using the client.beta.threads.messages.list method.

Construct a list of dictionaries, result, where each dictionary represents a message.

Each message dictionary contains the following keys:
* content: The actual text of the message.
* role: The role of the entity that sent the message.
* hidden: A boolean indicating whether the message is hidden or not.
* id: The unique identifier of the message.
* created_at: The timestamp when the message was created.

Returns the result list.
***
