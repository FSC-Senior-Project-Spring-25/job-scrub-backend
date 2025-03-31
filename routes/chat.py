from fastapi import APIRouter, Depends, Form, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Optional

from dependencies import LLM, Parser, get_current_user
from models.chat import ChatMessage

router = APIRouter()


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
        llm: LLM,
        parser: Parser,
        message: str = Form(...),
        files: Optional[List[UploadFile]] = File(None),
        current_user: dict = Depends(get_current_user),
):
    """
    Stream chat responses with optional file context

    Args:
        message: User's message
        files: Optional list of context files (PDFs)
        llm: LLM service
        parser: PDF parser service
        current_user: Authenticated user info
    """
    try:
        # Process any context files
        context = ""
        if files:
            for file in files:
                if file.filename.endswith('.pdf'):
                    file_bytes = await file.read()
                    context += f"\nDocument content:\n{parser.parse_pdf(file_bytes)}"

        # Create system prompt
        system_prompt = (
            "You are a helpful AI assistant. Respond concisely and professionally. "
            "If provided with document context, use it to inform your responses."
        )

        if context:
            system_prompt += context

        return StreamingResponse(
            generate_chat_response(llm, system_prompt, message),
            media_type="text/event-stream"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("")
async def chat(
        chat_message: ChatMessage,
        llm: LLM,
        current_user: dict = Depends(get_current_user),
):
    """
    Non-streaming chat endpoint for simple requests

    Args:
        chat_message: Message content and optional context files
        llm: LLM service
        current_user: Authenticated user info
    """
    try:
        system_prompt = (
            "You are a helpful AI assistant. "
            "Respond concisely and professionally."
        )

        response = await llm.generate(system_prompt, chat_message.content)
        if not response.success:
            raise HTTPException(status_code=500, detail=response.error)

        return {"message": response.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
