from app.models.base import Base
from sqlalchemy import Column, String, UUID, JSON, LargeBinary


class UserDocument(Base):
    __tablename__ = 'user_document'

    id = Column(UUID(as_uuid=True), primary_key=True)
    content = Column(LargeBinary)
    ocr_data = Column(JSON)
    session_id = Column(UUID(as_uuid=True))
    type = Column(String)
