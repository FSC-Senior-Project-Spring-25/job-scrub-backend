from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Message:
    """Message in the conversation history"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
