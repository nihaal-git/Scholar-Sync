"""
Auth API router — FastAPI endpoints for user registration, login, and management.

Features:
- POST /auth/register  — create a new account
- POST /auth/login     — authenticate and receive JWT
- POST /auth/logout    — clear auth cookie
- GET  /auth/me        — get current user info (protected)
- POST /auth/change-password — change password (protected)

Rate limiting on login to prevent brute-force attacks.
JWT delivered via HTTP-only cookie for security.
"""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Request, Response, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from scholarsync.auth.models import (
    RegisterRequest,
    LoginRequest,
    TokenResponse,
    UserResponse,
    ChangePasswordRequest,
)
from scholarsync.auth import service
from scholarsync.auth.security import create_access_token, decode_access_token
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# ── Rate Limiting (in-memory) ───────────────────────────────────────
# Tracks login attempts per IP: {ip: [(timestamp, ...), ...]}
_login_attempts: dict[str, list[float]] = defaultdict(list)
_MAX_ATTEMPTS = 5       # Max attempts per window
_WINDOW_SECONDS = 60    # Window duration


def _check_rate_limit(ip: str) -> None:
    """Raise 429 if the IP has exceeded the login attempt limit."""
    now = time.time()
    # Prune old entries
    _login_attempts[ip] = [t for t in _login_attempts[ip] if now - t < _WINDOW_SECONDS]
    if len(_login_attempts[ip]) >= _MAX_ATTEMPTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again in a minute.",
        )
    _login_attempts[ip].append(now)


# ── JWT Cookie Helper ───────────────────────────────────────────────

def _set_auth_cookie(response: Response, token: str) -> None:
    """Set JWT as a cookie readable by the frontend auth guard."""
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=False,  # Must be False so JS auth guard can check it
        samesite="lax",
        secure=False,  # Set True in production with HTTPS
        max_age=60 * 60 * 24,  # 24 hours
        path="/",
    )


def _clear_auth_cookie(response: Response) -> None:
    """Remove the JWT cookie."""
    response.delete_cookie(key="access_token", path="/")


# ── Dependency: Get Current User ────────────────────────────────────
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user_local(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict:
    """
    FastAPI dependency: extract and verify JWT from cookie or Authorization header.

    Priority: Authorization header > cookie.
    Returns user dict or raises 401.
    """
    token = None

    # 1. Try Authorization header
    if creds and creds.credentials:
        token = creds.credentials

    # 2. Fall back to cookie
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Please log in.",
        )

    # Decode JWT
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again.",
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload.",
        )

    # Fetch user from database
    user = await service.get_user_by_id(int(user_id))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found.",
        )

    return user


# ── POST /auth/register ─────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest, response: Response):
    """
    Register a new user account.

    On success, sets an HTTP-only auth cookie and returns the JWT.
    """
    user = await service.register_user(
        username=request.username,
        password=request.password,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists. Please choose a different username.",
        )

    # Create JWT
    token = create_access_token(
        data={"sub": str(user["id"]), "username": user["username"]},
    )

    # Set cookie
    _set_auth_cookie(response, token)

    logger.info("User registered: %s", user["username"])
    return TokenResponse(
        access_token=token,
        username=user["username"],
    )


# ── POST /auth/login ────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, response: Response, req: Request):
    """
    Authenticate a user and issue a JWT.

    Rate-limited to prevent brute-force attacks.
    On success, sets an HTTP-only auth cookie and returns the JWT.
    """
    # Rate limiting by client IP
    client_ip = req.client.host if req.client else "unknown"
    _check_rate_limit(client_ip)

    user = await service.authenticate_user(
        username=request.username,
        password=request.password,
    )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    # Create JWT
    token = create_access_token(
        data={"sub": str(user["id"]), "username": user["username"]},
    )

    # Set cookie
    _set_auth_cookie(response, token)

    logger.info("User logged in: %s", user["username"])
    return TokenResponse(
        access_token=token,
        username=user["username"],
    )


# ── POST /auth/logout ───────────────────────────────────────────────

@router.post("/logout")
async def logout(response: Response):
    """Log out by clearing the auth cookie."""
    _clear_auth_cookie(response)
    return {"message": "Logged out successfully."}


# ── GET /auth/me ─────────────────────────────────────────────────────

@router.get("/me", response_model=UserResponse)
async def get_me(user: dict = Depends(get_current_user_local)):
    """Return the currently authenticated user's information."""
    return UserResponse(**user)


# ── POST /auth/change-password ───────────────────────────────────────

@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    user: dict = Depends(get_current_user_local),
):
    """
    Change the current user's password.

    Requires the current password for verification.
    """
    success = await service.change_password(
        user_id=user["id"],
        current_password=request.current_password,
        new_password=request.new_password,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect.",
        )

    return {"message": "Password changed successfully."}
