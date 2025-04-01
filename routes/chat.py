import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, Form, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from starlette.responses import JSONResponse

from dependencies import LLM, Parser, get_current_user, SupervisorAgent

router = APIRouter()


async def process_files(files: List[UploadFile], parser: Parser) -> List[Dict[str, Any]]:
    """
    Process uploaded files and extract content

    Args:
        files: List of uploaded files
        parser: Parser service to extract content

    Returns:
        List of processed file data
    """
    processed_files = []

    for file in files:
        file_bytes = await file.read()
        file_type = "text"
        content = None

        if file.filename.endswith('.pdf'):
            file_type = "pdf"
            content = parser.parse_pdf(file_bytes)

        processed_files.append({
            "filename": file.filename,
            "type": file_type,
            "bytes": file_bytes,
            "content": content
        })

    return processed_files


async def generate_chat_response(
        llm: LLM,
        system_prompt: str,
        message: str
):
    """Stream chat response chunks"""
    try:
        async for chunk in llm.generate_stream(system_prompt, message):
            if chunk:
                yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: Error: {str(e)}\n\n"


@router.post("/stream")
async def chat_stream(
        parser: Parser,
        supervisor: SupervisorAgent,
        message: str = Form(...),
        files: Optional[List[UploadFile]] = File(None),
        conversation_history: str = Form("[]"),
        current_user: dict = Depends(get_current_user),
):
    """
    Stream chat responses with optional file context
    """
    try:
        print(f"[CHAT_STREAM] Processing message: {message[:50]}...")
        print(f"[CHAT_STREAM] User: {current_user['user_id']}")
        print(f"[CHAT_STREAM] Files: {len(files) if files else 0}")

        # Parse conversation history
        history = json.loads(conversation_history)
        print(f"[CHAT_STREAM] History items: {len(history)}")

        # Process any uploaded files
        processed_files = await process_files(files or [], parser)
        print(f"[CHAT_STREAM] Processed files: {len(processed_files)}")

        # Process the message through supervisor
        print(f"[CHAT_STREAM] Calling supervisor.process_message")
        result = await supervisor.process_message(
            user_id=current_user["user_id"],
            message=message,
            conversation_history=history,
            files=processed_files if processed_files else None
        )

        print(f"[CHAT_STREAM] Supervisor selected agent: {result['selected_agent']}")
        print(f"[CHAT_STREAM] Response: {result['response'][:100]}..." if result["response"] else "No response")

        return JSONResponse(
            content=result
        )
    except Exception as e:
        print(f"[CHAT_STREAM] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return StreamingResponse(
            iter([f"data: Error: {str(e)}\n\n"]),
            media_type="text/event-stream"
        )


@router.post("/message")
async def process_message(
        supervisor: SupervisorAgent,
        parser: Parser,
        message: str = Form(...),
        conversation_id: Optional[str] = Form(None),
        conversation_history: str = Form("[]"),
        files: Optional[List[UploadFile]] = File(None),
        current_user: dict = Depends(get_current_user),
):
    """
    Process a user message through the supervisor agent

    Args:
        message: User's message text
        conversation_id: Optional conversation identifier
        conversation_history: Previous conversation in JSON format
        files: Optional files attached to the message
        supervisor: Supervisor agent
        parser: PDF parser service
        current_user: Authenticated user info

    Returns:
        Agent response and updated conversation history
    """
    try:
        # Parse conversation history
        history = json.loads(conversation_history)

        # Process any uploaded files
        processed_files = await process_files(files or [], parser)

        # Process the message
        result = await supervisor.process_message(
            user_id=current_user["user_id"],
            message=message,
            conversation_history=history,
            files=processed_files if processed_files else None
        )

        return {
            "response": result["response"],
            "conversation": result["conversation"],
            "conversation_id": conversation_id or f"conv_{datetime.now().timestamp()}",
            "selected_agent": result["selected_agent"],
            "error": result["error"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations")
async def get_conversations(
        current_user: dict = Depends(get_current_user)
):
    """
    Get list of user's conversations

    Args:
        current_user: Authenticated user info

    Returns:
        List of conversation summaries
    """
    # This would typically fetch from a database
    # For now, return an empty list as placeholder
    return {"conversations": []}


@router.get("/conversations/{conversation_id}")
async def get_conversation(
        conversation_id: str,
        current_user: dict = Depends(get_current_user)
):
    """
    Get a specific conversation history

    Args:
        conversation_id: Conversation identifier
        current_user: Authenticated user info

    Returns:
        Complete conversation history
    """
    # This would typically fetch from a database
    # For now, return an empty conversation as placeholder
    return {
        "conversation_id": conversation_id,
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
