from typing import List, Dict, Optional, Any

from models.chat import Message
from services.llm.base.llm import LLM, ResponseFormat


async def handle_chat(
        llm: LLM,
        message: str,
        conversation_history: List[Message],
        files: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Handle general chat messages using Gemini

    Args:
        llm: LLM instance to use for generation
        message: User's current message
        conversation_history: Previous conversation messages
        files: Optional attached files

    Returns:
        Assistant response
    """
    # Debug info
    print(f"[CHAT_HANDLER] Processing message: {message[:50]}...")
    print(f"[CHAT_HANDLER] Files: {len(files) if files else 0}")
    print(f"[CHAT_HANDLER] History items: {len(conversation_history)}")

    system_prompt_template = (
        "You are an AI assistant specialized in career development, job searching, "
        "and resume improvement. Be helpful, concise, and professional. "
        "{context}"
    )
    try:
        # Extract file context if any files are attached
        file_context = ""
        if files:
            file_context = "File context:\n"
            for file in files:
                filename = file.get("filename", "unnamed file")
                filetype = file.get("type", "unknown")

                if filetype == "text" and "content" in file:
                    file_context += f"\n--- Content from {filename} ---\n{file['content']}\n"
                elif filetype == "pdf" and "content" in file:
                    # Assuming PDF content is already extracted as text
                    file_context += f"\n--- Content from PDF {filename} ---\n{file['content']}\n"
                else:
                    file_context += f"\n(File attached: {filename}, but content is not accessible)\n"

        # Format conversation history
        formatted_history = []
        for msg in conversation_history[-5:]:  # Use last 5 messages for context
            formatted_history.append(f"{msg.role.upper()}: {msg.content}")

        conversation_context = "\n".join(formatted_history) if formatted_history else ""

        # Create system prompt with context
        context_str = ""
        if file_context:
            context_str = f"The user has shared the following files: {file_context}"

        system_prompt = system_prompt_template.format(context=context_str)

        # Generate response
        user_prompt = f"""
            {conversation_context}

            USER: {message}
            """

        # Debug the prompt
        print(f"[CHAT_HANDLER] System prompt: {system_prompt[:100]}...")
        print(f"[CHAT_HANDLER] User prompt: {user_prompt[:100]}...")

        if not llm:
            return "Error: LLM instance not provided to chat handler"

        response = await llm.agenerate(
            system_prompt=system_prompt,
            user_message=user_prompt,
            response_format=ResponseFormat.RAW
        )

        if not response.success:
            return f"I'm sorry, I encountered an error while processing your message. {response.error}"

        print(f"[CHAT_HANDLER] Generated response: {response.content[:100]}...")
        return response.content

    except Exception as e:
        print(f"[CHAT_HANDLER] Error: {str(e)}")
        return f"I apologize, but I encountered an error while processing your request: {str(e)}"
