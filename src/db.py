"""
Shared DB connection helper.

Defaults to the local SQLite database (data/olist.db). If a DATABASE_URL
env var is set, connects to that instead
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent.parent / "data" / "olist.db"


def get_engine():
    """Return a SQLAlchemy engine: cloud Postgres if DATABASE_URL is set, else local SQLite."""
  

    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return create_engine(database_url)
    return create_engine(f"sqlite:///{DEFAULT_SQLITE_PATH}")
