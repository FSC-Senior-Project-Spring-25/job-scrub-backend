from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from pydantic import BaseModel


class ChatMessage(BaseModel):
    content: str
    context_files: Optional[List[str]] = None

@dataclass
class Message:
    """Message in the conversation history"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)