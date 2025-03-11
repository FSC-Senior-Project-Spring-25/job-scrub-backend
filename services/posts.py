from fastapi import APIRouter, HTTPException, Depends
from dependencies import FirestoreDB, get_db, get_current_user
from models.post import Post
from datetime import datetime
from typing import List
from dependencies import get_db, FirestoreDB
from google.cloud import firestore
import bleach

router = APIRouter()

# Get all posts
@router.get("/posts", response_model=List[Post])
def get_posts(db: FirestoreDB = Depends(get_db)):
    posts_ref = db.collection("posts").order_by("created_at", direction=firestore.Query.DESCENDING).stream()
    posts = []
    for doc in posts_ref:
        post_data = doc.to_dict()
        post_data["id"] = doc.id  # attach the document ID if needed
        posts.append(post_data)
    return posts

@router.post("/posts")
def create_post(author: str, content: str, db: FirestoreDB = Depends(get_db)):
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
def like_post(post_id: str, db: FirestoreDB = Depends(get_db)):
    post_ref = db.collection("posts").document(post_id)
    snapshot = post_ref.get()

    if not snapshot.exists:
        raise HTTPException(status_code=404, detail="Post not found")

    post_data = snapshot.to_dict()
    current_likes = post_data.get("likes", 0)
    post_ref.update({"likes": current_likes + 1})

    return {"message": "Post liked", "new_likes": current_likes + 1}


@router.delete("/posts/{post_id}/likes")
def remove_like(post_id: str, user_id: str, db: FirestoreDB = Depends()):
    post_ref = db.collection("posts").document(post_id)
    snapshot = post_ref.get()
    if not snapshot.exists:
        raise HTTPException(status_code=404, detail="Post not found")
    
    like_ref = post_ref.collection("likes").document(user_id)
    if not like_ref.get().exists:
        raise HTTPException(status_code=400, detail="User hasn't liked this post")
    
@router.delete("/posts/{post_id}/likes")
def remove_like(post_id: str, user_id: str, db: FirestoreDB = Depends()):
    post_ref = db.collection("posts").document(post_id)
    snapshot = post_ref.get()
    if not snapshot.exists:
        raise HTTPException(status_code=404, detail="Post not found")
    
    like_ref = post_ref.collection("likes").document(user_id)
    if not like_ref.get().exists:
        raise HTTPException(status_code=400, detail="User hasn't liked this post")
    
    like_ref.delete()
    
    post_data = snapshot.to_dict()
    current_like_count = post_data.get("likeCount", 0)
    if current_like_count > 0:
        post_ref.update({"likeCount": current_like_count - 1})
    
    return {"message": f"User {user_id} unliked post {post_id}"}

#  Add a comment to a post
@router.post("/posts/{post_id}/comment")
def add_comment(post_id: str, author: str, text: str, db: FirestoreDB = Depends(get_db)):
    post_ref = db.collection("posts").document(post_id)
    if not post_ref.get().exists:
        raise HTTPException(status_code=404, detail="Post not found")

    comment_ref = post_ref.collection("comments").document()  # Auto-generate comment ID
    sanitized_text = bleach.clean(text, strip=True)
    comment_data = {
        "author": author,  # UID of the commenter
        "text": sanitized_text,
        "created_at": datetime.utcnow().isoformat()
    }
    comment_ref.set(comment_data)
    return {"message": "Comment added", "comment_id": comment_ref.id}

@router.get("/posts/{post_id}/comments")
def get_comments(post_id: str, db: FirestoreDB = Depends()):
    post_ref = db.collection("posts").document(post_id)
    if not post_ref.get().exists:
        raise HTTPException(status_code=404, detail="Post not found")
    comments_ref = post_ref.collection("comments") \
                           .order_by("created_at", direction=firestore.Query.DESCENDING) \
                           .stream()
    comments = []
    for doc in comments_ref:
        c_data = doc.to_dict()
        c_data["id"] = doc.id
        comments.append(c_data)
    return comments

