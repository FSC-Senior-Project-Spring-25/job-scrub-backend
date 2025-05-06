from typing import Optional, List

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[int] = None
    resumeFile: Optional[dict] = None
    activeAgents: Optional[List[str]] = None

class ConversationCreate(BaseModel):
    firstMessage: str
    messages: List[Message]

class ConversationUpdate(BaseModel):
    messages: List[Message]

class Conversation(BaseModel):
    id: str
    firstMessage: str
    lastMessageTimestamp: int
    messages: List[Message]
