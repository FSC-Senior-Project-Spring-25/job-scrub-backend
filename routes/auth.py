
from fastapi import APIRouter, Request, Response, HTTPException
from firebase_admin import auth

from models.token import TokenRequest

router = APIRouter()


@router.post("/login")
async def login(request: TokenRequest, response: Response):
    try:
        # Verify the ID token
        decoded_token = auth.verify_id_token(request.id_token)

        # Create a session cookie
        expires_in = 5 * 24 * 60 * 60  # 5 days in seconds
        session_cookie = auth.create_session_cookie(
            request.id_token,
            expires_in=expires_in
        )

        # Set the cookie in the response
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.set_cookie(
            key="session",
            value=str(session_cookie),
            httponly=True,
            secure=False,  # TODO change when deployed
            max_age=expires_in,
            path="/",
            samesite="lax",
            domain=None
        )

        return {"success": True, "user_id": decoded_token["uid"]}
    except Exception as e:
        print(f"Error in login: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


@router.post("/logout")
async def logout(response: Response):
    # Clear the session cookie
    response.delete_cookie(
        key="session",
        path="/",
        httponly=True,
        secure=False  # TODO change when deployed
    )
    return {"success": True}


@router.get("/verify")
async def verify_session(request: Request):
    try:
        # Get the session cookie
        session_cookie = request.cookies.get("session")

        if not session_cookie:
            raise HTTPException(status_code=401, detail="No session cookie found")

        # Verify the session cookie
        decoded_claims = auth.verify_session_cookie(
            session_cookie,
            check_revoked=True
        )

        # Session is valid
        return {
            "valid": True,
            "user": {
                "uid": decoded_claims["uid"],
                "email": decoded_claims.get("email")
            }
        }
    except auth.InvalidSessionCookieError:
        raise HTTPException(status_code=401, detail="Invalid session")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Session verification failed: {str(e)}")


