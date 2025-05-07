from datetime import datetime

import bleach
import firebase_admin
from firebase_admin import firestore as fs
from google.cloud import firestore
from typing import List, Dict, Any, Optional, Set

from google.cloud.firestore_v1 import FieldFilter


class FirestoreDB:
    def __init__(self, app: firebase_admin.App):
        self.db = fs.client(app)

    def collection(self, name: str):
        return self.db.collection(name)

    def get_all_posts(self):
        """Get all posts sorted by creation date descending"""
        posts_ref = self.collection("posts").order_by("created_at", direction=firestore.Query.DESCENDING).stream()
        posts = []
        for doc in posts_ref:
            post_data = doc.to_dict()
            post_data["id"] = doc.id
            posts.append(post_data)
        return posts

    def get_posts_with_user_data(self, user_id: str):
        """
        Get all posts with user-specific like status and comments in a single efficient operation
        """
        # Get all posts
        posts = self.get_all_posts()

        if not posts:
            return []

        # Extract all post IDs
        post_ids = [post["id"] for post in posts]

        # Batch fetch likes and comments for all posts
        user_liked_posts = self.get_user_likes_for_posts(post_ids, user_id)
        all_comments = self.get_comments_for_posts(post_ids)

        # Join the data in memory
        for post in posts:
            post_id = post["id"]
            post["userHasLiked"] = post_id in user_liked_posts
            post["comments"] = all_comments.get(post_id, [])

        return posts

    def get_user_likes_for_posts(self, post_ids: List[str], user_id: str) -> Set[str]:
        """
        Batch fetch all posts that a user has liked from a list of post IDs
        Returns a set of post IDs that the user has liked
        """
        if not post_ids or not user_id:
            return set()

        # Instead of checking subcollections, query the flattened 'likes' collection
        liked_posts = set()

        # Process in chunks of 10 due to Firestore 'in' query limitation
        for i in range(0, len(post_ids), 10):
            chunk = post_ids[i:i + 10]

            likes_ref = self.collection("likes").where(
                filter=FieldFilter("user_id", "==", user_id)
            ).where(
                filter=FieldFilter("post_id", "in", chunk)
            ).stream()

            # Add all liked post IDs to the set
            for doc in likes_ref:
                like_data = doc.to_dict()
                liked_posts.add(like_data.get("post_id"))

        return liked_posts

    def get_comments_for_posts(self, post_ids: List[str]) -> Dict[str, List[Dict]]:
        """
        Batch fetch all comments for multiple posts
        Returns a dictionary mapping post_ids to lists of comments
        """
        if not post_ids:
            return {}

        # Initialize result dictionary
        comments_by_post = {post_id: [] for post_id in post_ids}

        # Process in chunks of 10 due to Firestore 'in' query limitation
        for i in range(0, len(post_ids), 10):
            chunk = post_ids[i:i + 10]

            comments_ref = self.collection("comments").where(
                filter=FieldFilter("post_id", "in", chunk)
            ).order_by(
                "created_at", direction=firestore.Query.DESCENDING
            ).stream()

            # Group comments by post_id
            for doc in comments_ref:
                comment = doc.to_dict()
                comment["id"] = doc.id
                post_id = comment.get("post_id")
                if post_id in comments_by_post:
                    comments_by_post[post_id].append(comment)

        return comments_by_post

    def create_post(self, author: str, author_uid: str, content: str):
        """Create a new post"""
        new_post_ref = self.collection("posts").document()
        new_post_data = {
            "author": author,
            "author_uid": author_uid,
            "content": content,
            "created_at": datetime.now().isoformat(),
            "likes": 0
        }
        new_post_ref.set(new_post_data)
        return new_post_ref.id

    def get_post(self, post_id: str):
        """Get a post by ID"""
        post_ref = self.collection("posts").document(post_id)
        snapshot = post_ref.get()
        if not snapshot.exists:
            return None
        return snapshot

    def update_post_likes(self, post_id: str, increment: int = 1):
        """Update the like count for a post"""
        post_ref = self.collection("posts").document(post_id)
        transaction = self.db.transaction()

        @firestore.transactional
        def update_in_transaction(transaction, post_ref):
            snapshot = post_ref.get(transaction=transaction)
            if not snapshot.exists:
                return None

            post_data = snapshot.to_dict()
            current_likes = post_data.get("likes", 0)
            new_likes = current_likes + increment

            # Update atomically within the transaction
            transaction.update(post_ref, {"likes": new_likes})
            return new_likes

        return update_in_transaction(transaction, post_ref)

    def add_like(self, post_id: str, user_id: str):
        """Add a like to a post from a specific user"""
        # First check if the like already exists
        like_id = f"{post_id}_{user_id}"
        like_ref = self.collection("likes").document(like_id)

        if like_ref.get().exists:
            # Like already exists, don't increment counter again
            return None

        # Add the like document
        like_data = {
            "post_id": post_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        like_ref.set(like_data)

        # Update post like count
        return self.update_post_likes(post_id, 1)

    def remove_like(self, post_id: str, user_id: str):
        """Remove a like from a post for a specific user"""
        like_id = f"{post_id}_{user_id}"
        like_ref = self.collection("likes").document(like_id)

        if not like_ref.get().exists:
            return None

        # Delete the like document
        like_ref.delete()

        # Update post like count
        return self.update_post_likes(post_id, -1)

    def user_has_liked_post(self, post_id: str, user_id: str) -> bool:
        """Check if a specific user has liked a post"""
        like_id = f"{post_id}_{user_id}"
        like_ref = self.collection("likes").document(like_id)
        return like_ref.get().exists

    def add_comment(self, post_id: str, author: str, author_id: str, text: str):
        """Add a comment to a post"""
        comment_ref = self.collection("comments").document()
        sanitized_text = bleach.clean(text, strip=True)
        comment_data = {
            "post_id": post_id,
            "author": author,
            "author_uid": author_id,
            "text": sanitized_text,
            "created_at": datetime.now().isoformat()
        }
        comment_ref.set(comment_data)
        return comment_ref.id

    def get_comments(self, post_id: str):
        """Get comments for a post"""
        comments_ref = self.collection("comments").where(
            filter=FieldFilter("post_id", "==", post_id)
        ).order_by(
            "created_at", direction=firestore.Query.DESCENDING
        ).stream()

        comments = []
        for doc in comments_ref:
            c_data = doc.to_dict()
            c_data["id"] = doc.id
            comments.append(c_data)
        return comments

    def search_users(self, q: str):
        """
        Fetch all users and return those whose email or username
        contains the query string (case-insensitive).
        """
        q_lower = q.strip().lower()
        users_ref = self.collection("users").stream()

        results = []
        for doc in users_ref:
            data = doc.to_dict()
            email = data.get("email", "") or ""
            username = data.get("username", "") or ""
            # case-insensitive substring match
            if q_lower in email.lower() or q_lower in username.lower():
                # include the full profile
                results.append({"id": doc.id, **data})

        return results

    def create_conversation(self, user_id: str, first_message: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a new conversation for a user"""
        conversation_ref = self.collection("conversations").document()
        timestamp = datetime.now().timestamp() * 1000  # Convert to milliseconds

        conversation_data = {
            "user_id": user_id,
            "firstMessage": first_message,
            "lastMessageTimestamp": int(timestamp),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": messages
        }

        conversation_ref.set(conversation_data)

        # Return the conversation with its ID
        return {
            "id": conversation_ref.id,
            "firstMessage": first_message,
            "lastMessageTimestamp": int(timestamp),
            "messages": messages
        }

    def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID for a specific user"""
        conversation_ref = self.collection("conversations").document(conversation_id)
        snapshot = conversation_ref.get()

        if not snapshot.exists:
            return None

        conversation_data = snapshot.to_dict()

        # Check if this conversation belongs to the user
        if conversation_data.get("user_id") != user_id:
            return None

        # Format the response
        return {
            "id": snapshot.id,
            "firstMessage": conversation_data.get("firstMessage", ""),
            "lastMessageTimestamp": conversation_data.get("lastMessageTimestamp", 0),
            "messages": conversation_data.get("messages", [])
        }

    def update_conversation(self, conversation_id: str, user_id: str, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Update a conversation's messages"""
        conversation_ref = self.collection("conversations").document(conversation_id)
        snapshot = conversation_ref.get()

        if not snapshot.exists:
            return None

        conversation_data = snapshot.to_dict()

        # Check if this conversation belongs to the user
        if conversation_data.get("user_id") != user_id:
            return None

        # Update the conversation
        timestamp = datetime.now().timestamp() * 1000  # Convert to milliseconds
        conversation_ref.update({
            "messages": messages,
            "lastMessageTimestamp": int(timestamp),
            "updated_at": datetime.now().isoformat()
        })

        # Return the updated conversation
        return {
            "id": conversation_id,
            "firstMessage": conversation_data.get("firstMessage", ""),
            "lastMessageTimestamp": int(timestamp),
            "messages": messages
        }

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation"""
        conversation_ref = self.collection("conversations").document(conversation_id)
        snapshot = conversation_ref.get()

        if not snapshot.exists:
            return False

        conversation_data = snapshot.to_dict()

        # Check if this conversation belongs to the user
        if conversation_data.get("user_id") != user_id:
            return False

        # Delete the conversation
        conversation_ref.delete()
        return True

    def get_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all conversations for a user"""
        conversations_ref = self.collection("conversations") \
            .where(filter=FieldFilter("user_id", "==", user_id)) \
            .order_by("lastMessageTimestamp", direction=firestore.Query.DESCENDING) \
            .stream()

        conversations = []
        for doc in conversations_ref:
            conversation_data = doc.to_dict()
            conversations.append({
                "id": doc.id,
                "firstMessage": conversation_data.get("firstMessage", ""),
                "lastMessageTimestamp": conversation_data.get("lastMessageTimestamp", 0),
                "messages": conversation_data.get("messages", [])
            })

        return conversations
