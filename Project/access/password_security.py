"""Turning a password and its salt into the form the database stores.

PBKDF2 runs the configured hash many thousands of times, so guessing passwords from a stolen
database is slow. Each account has its own random salt, so two people with the same password
still get different stored hashes and one cracked hash opens nothing else.
"""

import hashlib

from config import config


def hash_password(plain_password: str, password_salt: str) -> str:
    """The stored form of a password: a PBKDF2 key, written as hexadecimal text."""
    derived_key = hashlib.pbkdf2_hmac(
        config.PASSWORD_HASH_ALGORITHM,
        plain_password.encode("utf-8"),
        password_salt.encode("utf-8"),
        config.PASSWORD_HASH_ROUNDS,
    )
    return derived_key.hex()
