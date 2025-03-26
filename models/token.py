from pydantic import BaseModel


class TokenRequest(BaseModel):
    id_token: str
