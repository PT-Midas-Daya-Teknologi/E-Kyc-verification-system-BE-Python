from app.models.base import Base
from sqlalchemy import Column, Integer, UUID, JSON, Boolean, DateTime, String

class UserVideo(Base):
    __tablename__ = 'user_video'

    id = Column(UUID, primary_key=True)
    name = Column(String)
    path = Column(String)
    created_at = Column(DateTime)
    created_by = Column(String)
    updated_at = Column(DateTime)
    updated_by = Column(String)
    is_active = Column(Boolean)

    def __init__(self, id, name, path, created_at, created_by, updated_at, updated_by, is_active):
        self.id = id
        self.name = name
        self.path = path
        self.created_at = created_at
        self.created_by = created_by
        self.updated_at = updated_at
        self.updated_by = updated_by
        self.is_active = is_active

    def __repr__(self):
        return f"<UserVideo id={self.id} name={self.name}>"