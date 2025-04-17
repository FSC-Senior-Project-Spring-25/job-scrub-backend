
from fastapi import APIRouter, HTTPException, Depends
from google.cloud import firestore
from dependencies import Firestore
router = APIRouter()

@router.post("/{user_id}/follow")
def follow_user(user_id: str, target_id: str, db: Firestore):
    """
    Allows user 'user_id' to follow 'target_id'.
    Creates two subcollection documents:
      - In the user's 'following' subcollection for target_id.
      - In the target's 'followers' subcollection for user_id.
    """
    user_ref = db.collection("users").document(user_id)
    target_ref = db.collection("users").document(target_id)

    if not user_ref.get().exists:
        raise HTTPException(status_code=404, detail="User not found")
    if not target_ref.get().exists:
        raise HTTPException(status_code=404, detail="Target user not found")
    if user_id == target_id:
        raise HTTPException(status_code=400, detail="Cannot follow yourself")

    # Check if already following
    following_ref = user_ref.collection("following").document(target_id)
    if following_ref.get().exists:
        raise HTTPException(status_code=400, detail="Already following this user")

    # Create follow documents in both subcollections
    following_ref.set({"dateFollowed": firestore.SERVER_TIMESTAMP})
    target_ref.collection("followers").document(user_id).set({"dateFollowed": firestore.SERVER_TIMESTAMP})

    return {"message": f"{user_id} now follows {target_id}"}

@router.post("/{user_id}/unfollow")
def unfollow_user(user_id: str, target_id: str, db: Firestore):
    """
    Allows user 'user_id' to unfollow 'target_id'.
    Deletes follow documents from both the 'following' and 'followers' subcollections.
    """
    user_ref = db.collection("users").document(user_id)
    target_ref = db.collection("users").document(target_id)

    if not user_ref.get().exists or not target_ref.get().exists:
        raise HTTPException(status_code=404, detail="User not found")

    # Delete follow docs from both subcollections
    user_ref.collection("following").document(target_id).delete()
    target_ref.collection("followers").document(user_id).delete()

    return {"message": f"{user_id} unfollowed {target_id}"}

@router.get("/{user_id}/following")
def list_following(user_id: str, db: Firestore):
    """
    Returns a list of user IDs that 'user_id' is following.
    """
    user_ref = db.collection("users").document(user_id)
    if not user_ref.get().exists:
        raise HTTPException(status_code=404, detail="User not found")

    following_docs = user_ref.collection("following").stream()
    following_list = [doc.id for doc in following_docs]
    return {"following": following_list}

@router.get("/{user_id}/followers")
def list_followers(user_id: str, db: Firestore):
    """
    Returns a list of user IDs who follow 'user_id'.
    """
    user_ref = db.collection("users").document(user_id)
    if not user_ref.get().exists:
        raise HTTPException(status_code=404, detail="User not found")

    followers_docs = user_ref.collection("followers").stream()
    followers_list = [doc.id for doc in followers_docs]
    return {"followers": followers_list}
