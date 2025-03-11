from fastapi import APIRouter, HTTPException, Depends
from dependencies import FirestoreDB
from models.post import Post
from datetime import datetime
from typing import List
import bleach
router = APIRouter()

# Get all posts
@router.get("/posts", response_model=List[Post])
def get_posts(db: FirestoreDB):
    posts_ref = db.collection("posts").order_by("created_at", direction=firestore.Query.DESCENDING).stream()
    posts = [post.to_dict() for post in posts_ref]
    return posts
# Post Creation (Ep)
def create_post(author: str, content: str, db: FirestoreDB):
    new_post_ref = db.collection("posts").document()
    new_post_data = {
        "author": author,
        "content": content,
        "created_at": datetime.utcnow().isoformat(),
        "likes": 0,
        "comments": []
    }
    new_post_ref.set(new_post_data)
    return {"message": "Post created", "id": new_post_ref.id}
# Like a post
@router.post("/posts/{post_id}/like")
def like_post(post_id: str, db: FirestoreDB):
    post_ref = db.collection("posts").document(post_id)
    post = post_ref.get()

    if not post.exists:
        raise HTTPException(status_code=404, detail="Post not found")

    post_data = post.to_dict()
    post_ref.update({"likes": post_data["likes"] + 1})

    return {"message": "Post liked", "new_likes": post_data["likes"] + 1}

#  Add a comment to a post
@router.post("/posts/{post_id}/comment")
def add_comment(post_id: str, comment: str, db: FirestoreDB):
    post_ref = db.collection("posts").document(post_id)
    post = post_ref.get()

    if not post.exists:
        raise HTTPException(status_code=404, detail="Post not found")
    # This sanitizes comments using the bleach library (Ep)
    sanitized_comment = bleach.clean(comment, strip=True)
 
    post_data = post.to_dict()
    new_comments = post_data["comments"] + [sanitized_comment]
    post_ref.update({"comments": new_comments})

    return {"message": "Comment added", "total_comments": len(new_comments)}
