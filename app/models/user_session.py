from app.models.base import Base
from sqlalchemy import Column, Integer, UUID, JSON, Boolean, DateTime


class UserSession(Base):
    __tablename__ = 'user_session'

    id = Column(UUID, primary_key=True)
    attempts = Column(JSON)
    document_id = Column(UUID, foreign_key='document_id')
    is_active = Column(Boolean)
    session_expiry = Column(DateTime)
    user_id = Column(Integer, foreign_key='user_id')
    video_id = Column(UUID)