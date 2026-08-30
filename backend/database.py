from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load .env BEFORE reading the URL — rag_engine's own load_dotenv runs after
# this module is imported, so without this the API silently dropped to SQLite.
load_dotenv(override=True)

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./backend/database.db")

if (
    SQLALCHEMY_DATABASE_URL.startswith("sqlite")
    and ":memory:" not in SQLALCHEMY_DATABASE_URL
):
    _db_file = SQLALCHEMY_DATABASE_URL.split("///")[-1].split("?")[0]
    _db_dir = os.path.dirname(os.path.abspath(_db_file))
    os.makedirs(_db_dir, exist_ok=True)

if SQLALCHEMY_DATABASE_URL.startswith("postgres"):
    # Production: connection pooling for concurrent workers. With Neon/pgBouncer
    # in transaction mode, cap the pool small — pgbouncer manages the real pool.
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
        pool_pre_ping=True,
        pool_recycle=1800,
    )
else:
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
