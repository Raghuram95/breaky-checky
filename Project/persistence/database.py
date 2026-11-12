"""The database connection and the session factory, shared by the whole project.

Both objects are built when this module is first imported. Building an engine opens no
connection and creates nothing; the tables are created only by run_database_setup.py.
check_same_thread=False lets FastAPI's worker threads use connections from the shared pool.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import config

database_engine = create_engine(
    config.APPLICATION_DATABASE_URL, connect_args={"check_same_thread": False}
)

session_factory = sessionmaker(bind=database_engine)
