"""
auth.py
=======

Password hashing + JWT creation/verification for the Office Attendance
Management System backend. Kept as its own module (rather than inline in
main.py) so both ``main.py`` and any future service (e.g. a background
worker) can import the same auth primitives.

Dependencies:
    pip install "passlib[bcrypt]" "python-jose[cryptography]"

Known gotcha: passlib's bcrypt backend can raise
``AttributeError: module 'bcrypt' has no attribute '__about__'`` on some
bcrypt versions >=4.1. If you hit that, run:
    pip install "bcrypt==4.0.1"
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

# ---------------------------------------------------------------------------
# WARNING: this default secret is for local development ONLY. Before running
# this anywhere beyond your own laptop, replace it with a real secret, e.g.:
#     python -c "import secrets; print(secrets.token_hex(32))"
# and ideally load it from an environment variable instead of hardcoding it:
#     SECRET_KEY = os.environ["ATTENDANCE_JWT_SECRET"]
# ---------------------------------------------------------------------------
SECRET_KEY: str = "CHANGE_ME_BEFORE_DEPLOYING_-_generate_with_secrets.token_hex(32)"
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 12  # 12-hour login sessions

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Check a plaintext password against a stored bcrypt hash.

    Args:
        plain_password: The password the user just typed in.
        hashed_password: The bcrypt hash stored in ``Employee.hashed_password``.

    Returns:
        ``True`` if they match.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a plaintext password for storage.

    Args:
        password: Plaintext password (e.g. set during employee registration).

    Returns:
        A bcrypt hash safe to store in the database.
    """
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a signed JWT embedding whatever claims are passed in ``data``.

    Args:
        data: Claims to embed, e.g. ``{"employee_id": 3, "role": "admin"}``.
        expires_delta: Token lifetime. Defaults to ``ACCESS_TOKEN_EXPIRE_MINUTES``.

    Returns:
        The encoded JWT as a string.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """Decode + verify a JWT, returning its claims.

    Args:
        token: The raw JWT string (without the "Bearer " prefix).

    Returns:
        The decoded claims dict, or ``None`` if the token is missing,
        malformed, expired, or has an invalid signature.
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None
