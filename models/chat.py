from typing import Optional, List

from pydantic import BaseModel


class ChatMessage(BaseModel):
    content: str
    context_files: Optional[List[str]] = None