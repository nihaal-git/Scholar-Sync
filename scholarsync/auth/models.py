"""
Pydantic schemas for authentication — request/response models with validation.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ── Requests ─────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    """Request body for user registration."""
    username: str = Field(
        ...,
        min_length=3,
        max_length=30,
        description="Username (3-30 chars, alphanumeric + underscore only).",
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password (min 8 chars, must include uppercase, lowercase, and digit).",
    )
    confirm_password: str = Field(
        ...,
        description="Must match the password field.",
    )

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("Username can only contain letters, numbers, and underscores.")
        return v.lower()  # Store usernames as lowercase

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter.")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one digit.")
        return v

    @field_validator("confirm_password")
    @classmethod
    def validate_passwords_match(cls, v: str, info) -> str:
        password = info.data.get("password")
        if password and v != password:
            raise ValueError("Passwords do not match.")
        return v


class LoginRequest(BaseModel):
    """Request body for user login."""
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


# ── Responses ────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    """Response after successful authentication."""
    access_token: str
    token_type: str = "bearer"
    username: str


class UserResponse(BaseModel):
    """Public user information."""
    id: int
    username: str
    role: str
    created_at: str


class ChangePasswordRequest(BaseModel):
    """Request body for changing password."""
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password (min 8 chars).",
    )
    confirm_new_password: str = Field(
        ...,
        description="Must match the new password field.",
    )

    @field_validator("new_password")
    @classmethod
    def validate_new_password_strength(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter.")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one digit.")
        return v

    @field_validator("confirm_new_password")
    @classmethod
    def validate_new_passwords_match(cls, v: str, info) -> str:
        new_password = info.data.get("new_password")
        if new_password and v != new_password:
            raise ValueError("New passwords do not match.")
        return v
