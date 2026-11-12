"""Entry point 1: create the database, its six tables and the accounts people sign in with.

Safe to run again: tables are created only if missing and an account already present is left
exactly as it is.

Run from the Project folder:  python run_database_setup.py
"""

import os
import secrets

from access.password_security import hash_password
from config import config
from persistence.database import database_engine, session_factory
from persistence.models import Base, UserAccount
from persistence.repository import fetch_user_account, insert_user_account


def describe_account(account: dict) -> str:
    """One console line naming an account, its role and any agencies."""
    agencies = account["assigned_agencies"]
    role = account["role"] + (", agencies " + ", ".join(agencies) if agencies else "")
    return f"{account['username']:<8} {account['display_name']:<14} {role}"


def main() -> None:
    """Create the database folder, the tables and every seed account not already present."""
    os.makedirs(config.DATABASE_DIRECTORY, exist_ok=True)
    Base.metadata.create_all(database_engine)
    print(f"database: {config.APPLICATION_DATABASE_FILE}")
    print("tables: " + ", ".join(sorted(Base.metadata.tables)))

    created, already_present = 0, 0
    with session_factory() as session:
        for account in config.SEED_USER_ACCOUNTS:
            if fetch_user_account(session, UserAccount.username == account["username"]):
                already_present += 1
                continue
            password_salt = secrets.token_hex(config.PASSWORD_SALT_BYTES)
            insert_user_account(
                session,
                {
                    "username": account["username"],
                    "display_name": account["display_name"],
                    "role": account["role"],
                    "assigned_agencies": list(account["assigned_agencies"]),
                    "password_hash": hash_password(account["password"], password_salt),
                    "password_salt": password_salt,
                },
            )
            created += 1
            print("account created " + describe_account(account))
        session.commit()
    print(f"accounts created: {created}, already present: {already_present}")


if __name__ == "__main__":
    main()
