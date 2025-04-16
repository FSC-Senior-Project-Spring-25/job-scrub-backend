# routes/user_search.py
from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter()

@router.get("/search", tags=["user search"])
async def search_users(
    q: str = Query(..., min_length=1),
    request: Request
):
    """
    Search for users by name or email.
    """
    try:
        firestore_service = request.app.state.firestore
        results = firestore_service.search_users(q)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug", tags=["debug"])
async def debug_list_all_users(request: Request):
    """
    Return the first 50 user documents unfiltered,
    so we can inspect their shape and fields.
    """
    firestore_service = request.app.state.firestore
    docs = firestore_service.collection("users").limit(50).stream()
    users = [{"id": doc.id, **doc.to_dict()} for doc in docs]
    return {"count": len(users), "users": users}
