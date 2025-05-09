from sqlalchemy import Column, Integer, String
from .database import Base # type: ignore

class URL(Base):
    __tablename__ = "urls"

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, nullable=False)
    status = Column(String, nullable=False)