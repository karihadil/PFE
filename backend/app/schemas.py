from pydantic import BaseModel

class URLCreate(BaseModel):
    url: str
    status: str

class URLResponse(URLCreate):
    id: int

    class Config:
        orm_mode = True
