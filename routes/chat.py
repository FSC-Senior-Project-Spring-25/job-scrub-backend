import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Form, UploadFile, File
from fastapi.responses import StreamingResponse

from dependencies import Parser, CurrentUser, PineconeClient
from services.agents.supervisor_agent import SupervisorAgent
from services.agents.tools.create_supervisor_agent import create_supervisor_agent

router = APIRouter()


async def generate_stream_response(supervisor: SupervisorAgent, user_id: str, message: str):
    """Generate a streaming response from the supervisor agent"""
    try:
        async for chunk in supervisor.process_message(user_id=user_id, message=message):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


@router.post("")
async def chat(
        current_user: CurrentUser,
        parser: Parser,
        pinecone: PineconeClient,
        message: str = Form(...),
        resume: Optional[UploadFile] = File(None),
        conversation_history: str = Form("[]"),
):
    """
    Process a chat message and return a streaming response
    """
    try:
        print(f"[CHAT] Processing message: {message[:50]}...")
        print(f"[CHAT] User: {current_user.user_id}")
        print(f"[CHAT] Resume file: {resume.filename if resume else 'None'}")

        # Parse conversation history
        history = json.loads(conversation_history)
        print(f"[CHAT] History items: {len(history)}")

        # Create a new supervisor agent for this request
        supervisor = await create_supervisor_agent(
            user_id=current_user.user_id,
            pinecone_client=pinecone,
            conversation_history=history,
            resume_file=resume,
            resume_parser=parser
        )

        # Return a streaming response
        return StreamingResponse(
            generate_stream_response(
                supervisor=supervisor,
                user_id=current_user.user_id,
                message=message,
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
