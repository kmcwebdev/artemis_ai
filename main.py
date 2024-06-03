import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import AsyncOpenAI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput

load_dotenv()

app = FastAPI()

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
api_key = os.getenv("OPENAI_API_KEY")
functions = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the weather in location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["c", "f"]}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_employee_info",
            "description": "Get the employee information",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string", "description": "The employee id"},
                },
                "required": ["employee_id"]
            }
        }
    }
]

client = AsyncOpenAI(api_key=api_key)


class CreateAssistant(BaseModel):
    name: str
    instruction: str


class CreateMessage(BaseModel):
    assistant_id: str
    content: str


def get_current_weather(location, unit="f"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    elif "ph" in location.lower():
        return json.dumps({"location": "Phillipines", "temperature": "27", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def get_employee_info(employee_id):
    """Get the employee information"""
    if employee_id == "1":
        return json.dumps({"employee_id": employee_id, "name": "John", "age": 30})
    elif employee_id == "2":
        return json.dumps({"employee_id": employee_id, "name": "Jane", "age": 25})
    else:
        return json.dumps({"employee_id": employee_id, "name": "unknown", "age": "unknown"})


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/api/assistant/{assistant_id}")
async def get_assistant(assistant_id: str):
    assistant = await client.beta.assistants.retrieve(assistant_id=assistant_id)

    return {
        "assistant_id": assistant.id,
        "name": assistant.name,
        "instruction": assistant.instructions
    }


@app.post("/api/assistant")
async def new_assistant(data: CreateAssistant):
    assistant = await client.beta.assistants.create(
        name=data.name,
        instructions=data.instruction,
        tools=functions,
        temperature=0.5,
        model=llm,
    )

    return {
        "assistant_id": assistant.id
    }


@app.post("/api/threads/{assistant_id}")
async def new_session(assistant_id: str):
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


@app.get("/api/threads/{thread_id}/runs/{run_id}/status")
async def get_run(thread_id: str, run_id: str):
    run = await client.beta.threads.runs.retrieve(
        thread_id=thread_id,
        run_id=run_id
    )

    if (run.status == "requires_action"):
        tools_output = []
        tool_calls = run.required_action.submit_tool_outputs.tool_calls

        available_functions = {
            "get_current_weather": get_current_weather,
            "get_employee_info": get_employee_info
        }

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "get_current_weather":
                function_response = function_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit")
                )

            if function_name == "get_employee_info":
                function_response = function_to_call(
                    employee_id=function_args.get("employee_id")
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


@app.get("/api/threads/{thread_id}/messages")
async def get_thread(thread_id: str):
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


@app.post("/api/threads/{thread_id}/messages")
async def post_thread(thread_id: str, message: CreateMessage):
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
