import json
from fastapi import APIRouter, HTTPException
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
