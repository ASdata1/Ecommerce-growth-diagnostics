"""
Shared DB connection helper.

Connects to the local SQLite database (data/olist.db).
"""

from pathlib import Path

from sqlalchemy import create_engine

DEFAULT_SQLITE_PATH = Path(__file__).resolve().parent.parent / "data" / "olist.db"


def get_engine():
    """Return a SQLAlchemy engine for the local SQLite database."""
    return create_engine(f"sqlite:///{DEFAULT_SQLITE_PATH}")
