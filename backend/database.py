from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./backend/database.db")

if SQLALCHEMY_DATABASE_URL.startswith("sqlite") and ":memory:" not in SQLALCHEMY_DATABASE_URL:
    _db_file = SQLALCHEMY_DATABASE_URL.split("///")[-1].split("?")[0]
    _db_dir = os.path.dirname(os.path.abspath(_db_file))
    os.makedirs(_db_dir, exist_ok=True)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
