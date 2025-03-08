from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from .database import SessionLocal, engine, Base # type: ignore
from .models import URL # type: ignore
from .schemas import URLCreate, URLResponse # type: ignore

app = FastAPI()

# Create database tables
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/urls/", response_model=URLResponse)
def add_url(url_data: URLCreate, db: Session = Depends(get_db)):
    new_url = URL(url=url_data.url, status=url_data.status)
    db.add(new_url)
    db.commit()
    db.refresh(new_url)
    return new_url

@app.get("/urls/", response_model=list[URLResponse])
def get_urls(db: Session = Depends(get_db)):
    return db.query(URL).all()
