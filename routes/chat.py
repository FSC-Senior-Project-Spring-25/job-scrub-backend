import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Form, UploadFile, File
from fastapi.responses import StreamingResponse

from dependencies import LLM, Parser, CurrentUser, PineconeClient
from services.agents.tools.create_supervisor_agent import create_supervisor_agent

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


async def generate_stream_response(supervisor, user_id, message, history, files):
    """Generate a streaming response from the supervisor agent"""
    try:
        async for chunk in supervisor.process_message(
                user_id=user_id,
                message=message,
                conversation_history=history,
                files=files
        ):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


@router.post("")
async def chat(
        current_user: CurrentUser,
        parser: Parser,
        llm: LLM,
        pinecone: PineconeClient,
        message: str = Form(...),
        files: Optional[List[UploadFile]] = File(None),
        conversation_history: str = Form("[]"),
):
    """
    Process a chat message and return a streaming response
    """
    try:
        print(f"[CHAT] Processing message: {message[:50]}...")
        print(f"[CHAT] User: {current_user.user_id}")
        print(f"[CHAT] Files: {len(files) if files else 0}")

        # Parse conversation history
        history = json.loads(conversation_history)
        print(f"[CHAT] History items: {len(history)}")

        # Process any uploaded files
        processed_files = await process_files(files or [], parser)
        print(f"[CHAT] Processed files: {len(processed_files)}")

        # Create a new supervisor agent for this request
        supervisor = await create_supervisor_agent(
            user_id=current_user.user_id,
            pinecone_client=pinecone,
            llm=llm,
            conversation_history=history,
            files=processed_files if processed_files else None,
            resume_parser=parser
        )

        # Return a streaming response
        return StreamingResponse(
            generate_stream_response(
                supervisor,
                current_user.user_id,
                message,
                history,
                processed_files if processed_files else None
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        print(f"[CHAT] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return StreamingResponse(
            iter([f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"]),
            media_type="text/event-stream"
        )


@router.get("/conversations")
async def get_conversations(
        current_user: CurrentUser,
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
        current_user: CurrentUser,
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
