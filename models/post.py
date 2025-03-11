from pydantic import BaseModel
from typing import List, Optional

class Post(BaseModel):
    id: Optional[str]  # This field is optional (populated later)
    author: str
    content: str
    created_at: str
    likeCount: int = 0
    comments: List[str] = []
class LikeRequest(BaseModel):
    user_id: str
class CommentRequest(BaseModel):
    comment: str
