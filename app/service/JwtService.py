import os
import jwt
from fastapi import HTTPException

from app.models.user_session import UserSession
from app.service.Session import get_session


def validate_token(token: str):
    try:
        print(f"secret_key: {os.getenv("JWT_SECRET_KEY")}")
        # token = "Bearer " + token
        jwt_payload = jwt.decode(token, os.getenv("JWT_SECRET_KEY"), algorithms=["HS256"])

        session_id = jwt_payload["sub"]

        user_session_model = get_session().query(UserSession).filter(UserSession.id == session_id).first()
        if user_session_model is None:
            raise HTTPException(status_code=401, detail="Invalid session")

        return user_session_model

    except jwt.ExpiredSignatureError:
        # logger.error("Token expired")
        raise HTTPException(status_code=401, detail="Unauthorized")