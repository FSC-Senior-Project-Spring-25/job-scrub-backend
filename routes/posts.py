from datetime import datetime
from typing import List, Dict, Any

from fastapi import APIRouter

from dependencies import Firestore, CurrentUser
from models.post import Post, CommentRequest

router = APIRouter()


@router.get("")
async def get_posts(db: Firestore, current_user: CurrentUser) -> List[Dict[str, Any]]:
    """Get all posts with user-specific like status"""
    posts = db.get_all_posts()
    user_id = current_user.user_id

    # Fetch user's like status for each post
    for post in posts:
        # Check if this user has liked this post
        post_id = post["id"]
        user_like = db.user_has_liked_post(post_id, user_id)
        post["userHasLiked"] = user_like

        # Get comments for each post
        comments = db.get_comments(post_id)
        post["comments"] = comments or []

    return posts


@router.post("")
async def create_post(
        db: Firestore,
        post_data: Post,
        current_user: CurrentUser
) -> Dict[str, Any]:
    """Create a new post"""
    author = post_data.author
    author_uid = current_user.user_id
    post_id = db.create_post(author, author_uid, post_data.content)

    return {
        "id": post_id,
        "author": author,
        "author_uid": current_user.user_id,
        "content": post_data.content,
        "created_at": datetime.now().isoformat(),
        "likes": 0,
        "comments": []
    }


@router.post("/{post_id}/like")
async def toggle_like(
        db: Firestore,
        post_id: str,
        current_user: CurrentUser
) -> Dict[str, Any]:
    """Toggle like status for a post"""
    user_id = current_user.user_id

    # Check if user has already liked this post
    already_liked = db.user_has_liked_post(post_id, user_id)

    if already_liked:
        # Remove like
        new_like_count = db.remove_like(post_id, user_id)
        liked = False
    else:
        # Add like
        new_like_count = db.add_like(post_id, user_id)
        liked = True

    return {
        "post_id": post_id,
        "likes": new_like_count,
        "liked": liked
    }


@router.post("/{post_id}/comment")
async def add_comment(
        db: Firestore,
        post_id: str,
        comment: CommentRequest,
        current_user: CurrentUser
) -> Dict[str, Any]:
    """Add a comment to a post"""
    author = comment.author
    author_id = current_user.user_id
    comment_id = db.add_comment(post_id, author, author_id, comment.text)

    return {
        "id": comment_id,
        "post_id": post_id,
        "author": author,
        "author_uid": current_user.user_id,
        "text": comment.text,
        "created_at": datetime.now().isoformat()
    }
