import json
from typing import Optional, List

from fastapi import APIRouter, Form, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from dependencies import Parser, CurrentUser, PineconeClient, Firestore
from models.chat import Conversation, ConversationCreate, ConversationUpdate
from services.agents.supervisor_agent import SupervisorAgent
from services.agents.tools.create_supervisor_agent import create_supervisor_agent

router = APIRouter()


async def generate_stream_response(supervisor: SupervisorAgent, user_id: str, message: str):
    """Generate a streaming response from the supervisor agent"""

    # Variable to store the complete response for final update
    complete_response = ""

    try:
        async for chunk in supervisor.process_message(
                user_id=user_id, message=message
        ):
            # we only stream *assistant text* inside delta.content
            if chunk["type"] == "content_chunk":
                # Accumulate the full response
                complete_response += chunk["content"]
                payload = {"choices": [{"delta": {"content": chunk["content"]}}]}
            else:  # forward metadata once at the top
                payload = {"event": chunk["type"], **chunk}

                # If it's a completion event, make sure we include the full content
                if chunk["type"] == "complete" and "conversation" in chunk:
                    # Find and update the assistant message with complete_response
                    for msg in chunk["conversation"]:
                        if msg["role"] == "assistant" and msg["content"] is None:
                            msg["content"] = complete_response

            yield f"data: {json.dumps(payload)}\n\n"

        yield "data: [DONE]\n\n"
    except Exception as e:
        err = {"choices": [{"delta": {}}], "error": str(e)}
        yield f"data: {json.dumps(err)}\n\n"


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


@router.post("/conversations", response_model=Conversation)
async def create_conversation(
        data: ConversationCreate,
        current_user: CurrentUser,
        pinecone: PineconeClient,
        db: Firestore
):
    """Create a new conversation"""
    # Convert messages to dictionaries for storage
    messages_dict = [message.model_dump() for message in data.messages]

    # Create the conversation in Firestore
    conversation = db.create_conversation(
        user_id=current_user.user_id,
        first_message=data.firstMessage,
        messages=messages_dict
    )

    return conversation


@router.get("/conversations", response_model=List[Conversation])
async def get_conversations(
        current_user: CurrentUser,
        db: Firestore
):
    """Get all conversations for the current user"""
    conversations = db.get_user_conversations(current_user.user_id)
    return conversations


@router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(
        conversation_id: str,
        current_user: CurrentUser,
        db: Firestore
):
    """Get a specific conversation"""
    conversation = db.get_conversation(conversation_id, current_user.user_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.put("/conversations/{conversation_id}", response_model=Conversation)
async def update_conversation(
        data: ConversationUpdate,
        conversation_id: str,
        current_user: CurrentUser,
        db: Firestore
):
    """Update a conversation"""
    # Convert messages to dictionaries for storage
    messages_dict = [message.dict() for message in data.messages]

    conversation = db.update_conversation(
        conversation_id=conversation_id,
        user_id=current_user.user_id,
        messages=messages_dict
    )

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
        conversation_id: str,
        current_user: CurrentUser,
        db: Firestore
):
    """Delete a conversation"""
    success = db.delete_conversation(conversation_id, current_user.user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted successfully"}
