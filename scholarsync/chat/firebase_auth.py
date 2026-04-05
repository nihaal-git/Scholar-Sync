"""
Firebase Authentication middleware for FastAPI.

Verifies Firebase JWT tokens and manages user records in MongoDB.
"""

from __future__ import annotations

from datetime import datetime, timezone

import firebase_admin
from firebase_admin import auth as firebase_auth, credentials

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from scholarsync.config.settings import get_settings
from scholarsync.chat.database import get_db
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)

# ── Firebase Initialization ─────────────────────────────────────────
_firebase_app: firebase_admin.App | None = None


def init_firebase() -> None:
    """Initialize Firebase Admin SDK. Idempotent — safe to call multiple times."""
    global _firebase_app
    if _firebase_app is not None:
        return

    settings = get_settings()
    try:
        cred = credentials.Certificate(settings.firebase_credentials_path)
        _firebase_app = firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized")
    except Exception as e:
        logger.error("Failed to initialize Firebase: %s", e)
        raise RuntimeError(f"Firebase initialization failed: {e}") from e


# ── Security Scheme ─────────────────────────────────────────────────
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    """
    FastAPI dependency: verify Firebase JWT and return user dict.

    Flow:
    1. Extract Bearer token from Authorization header
    2. Verify with Firebase Admin SDK
    3. Upsert user in MongoDB (create on first login)
    4. Return user document

    Returns:
        dict with keys: _id (str), firebase_uid, email, created_at
    """
    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing. Provide a Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = creds.credentials

    # Verify the JWT
    try:
        decoded = firebase_auth.verify_id_token(token)
    except firebase_auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please re-authenticate.",
        )
    except firebase_auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )
    except Exception as e:
        logger.error("Token verification error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed.",
        )

    uid = decoded.get("uid")
    email = decoded.get("email", "")

    if not uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user identifier.",
        )

    # Upsert user in MongoDB
    db = get_db()
    user = await db.users.find_one_and_update(
        {"firebase_uid": uid},
        {
            "$set": {"email": email},
            "$setOnInsert": {
                "firebase_uid": uid,
                "created_at": datetime.now(timezone.utc),
            },
        },
        upsert=True,
        return_document=True,
    )

    # Return a clean user dict with string _id
    return {
        "_id": str(user["_id"]),
        "firebase_uid": user["firebase_uid"],
        "email": user.get("email", ""),
        "created_at": user.get("created_at"),
    }
