"""
database.py
===========

SQLAlchemy engine/session setup for the Office Attendance Management
System.

Note on the DB filename: earlier phases (``dashboard.py``,
``system_utilities.py``) were written assuming the database file is named
``attendance_system.db`` (not the original Todo-app ``sql_app.db``). This
module now matches that, so every part of the system reads/writes the
same file.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./attendance_system.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """FastAPI dependency: yields a DB session, always closed afterward."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
