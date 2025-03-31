from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    id_token: str = Field(..., alias="idToken")

