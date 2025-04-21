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

    # Verify users exist
    user_doc = user_ref.get()
    target_doc = target_ref.get()
    if not user_doc.exists:
        raise HTTPException(status_code=404, detail="User not found")
    if not target_doc.exists:
        raise HTTPException(status_code=404, detail="Target user not found")
    if user_id == target_id:
        raise HTTPException(status_code=400, detail="Cannot follow yourself")

    # Check if already following
    following_ref = user_ref.collection("following").document(target_id)
    if following_ref.get().exists:
        raise HTTPException(status_code=400, detail="Already following this user")

    # Get user data
    user_data = user_doc.to_dict()
    target_data = target_doc.to_dict()
    user_username = user_data.get("username", "Unknown")
    user_email = user_data.get("email", "Unknown")
    user_profile = user_data.get("profileIcon", "Unknown")

    # Get target user data
    target_username = target_data.get("username", "Unknown")
    target_email = target_data.get("email", "Unknown")
    target_profile = target_data.get("profileIcon", "Unknown")

    # Create follow documents in both subcollections with usernames
    following_ref.set({
        "dateFollowed": firestore.SERVER_TIMESTAMP,
        "username": target_username,
        "email": target_email,
        "profileIcon": target_profile
    })

    target_ref.collection("followers").document(user_id).set({
        "dateFollowed": firestore.SERVER_TIMESTAMP,
        "username": user_username,
        "email": user_email,
        "profileIcon": user_profile
    })

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
    Returns a list of users that 'user_id' is following, including their username.
    """
    user_ref = db.collection("users").document(user_id)
    if not user_ref.get().exists:
        raise HTTPException(status_code=404, detail="User not found")

    following_docs = user_ref.collection("following").stream()
    following_list = []

    for doc in following_docs:
        followed_id = doc.id
        followed_user = db.collection("users").document(followed_id).get()

        followed_user_dict = followed_user.to_dict()
        if followed_user.exists:
            following_list.append({
                "id": followed_id,
                "username": followed_user_dict.get("username", "Unknown"),
                "email": followed_user_dict.get("email", "Unknown"),
                "profileIcon": followed_user_dict.get("profileIcon")
            })


    return {"following": following_list}


@router.get("/{user_id}/followers")
def list_followers(user_id: str, db: Firestore):
    """
    Returns a list of users who follow 'user_id', including their username.
    """
    user_ref = db.collection("users").document(user_id)
    if not user_ref.get().exists:
        raise HTTPException(status_code=404, detail="User not found")

    followers_docs = user_ref.collection("followers").stream()
    followers_list = []

    for doc in followers_docs:
        follower_id = doc.id
        follower_user = db.collection("users").document(follower_id).get()

        followed_user_dict = follower_user.to_dict()
        if follower_user.exists:
            followers_list.append({
                "id": follower_id,
                "username": followed_user_dict.get("username", "Unknown"),
                "email": followed_user_dict.get("email", "Unknown"),
                "profileIcon": followed_user_dict.get("profileIcon")
            })

    return {"followers": followers_list}
