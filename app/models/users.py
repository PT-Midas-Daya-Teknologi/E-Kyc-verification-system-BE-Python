from app.models.base import Base
from sqlalchemy import Column, Integer, String

class Users(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    username = Column(String, unique=True)

    def __repr__(self):
        return f"<User(name='{self.name}', username='{self.username}')>"