from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
import base64

from fastapi import Depends, HTTPException, Request


AUTH_REQUIRED = os.environ.get("AUTH_REQUIRED", "0") == "1"
FIREBASE_ENABLED = os.environ.get("FIREBASE_ENABLED", "0") == "1"
API_KEY = os.environ.get("API_KEY")


_firebase_ready = False
try:
    if FIREBASE_ENABLED:
        import firebase_admin
        from firebase_admin import credentials, auth

        if not firebase_admin._apps:
            # Initialize with default credentials; use GOOGLE_APPLICATION_CREDENTIALS if set
            firebase_admin.initialize_app()
        _firebase_ready = True
except Exception:
    _firebase_ready = False


@dataclass
class AuthUser:
    uid: str
    email: Optional[str] = None


async def _verify_firebase_token(token: str) -> Optional[AuthUser]:
    if not _firebase_ready:
        return None
    try:
        from firebase_admin import auth

        decoded = auth.verify_id_token(token)
        return AuthUser(uid=decoded.get("uid"), email=decoded.get("email"))
    except Exception:
        return None


async def require_auth(request: Request) -> Optional[AuthUser]:
    if not AUTH_REQUIRED:
        return None

    # 1) Firebase Bearer token
    authz = request.headers.get("authorization") or request.headers.get("Authorization")
    if authz and authz.lower().startswith("bearer "):
        token = authz.split(" ", 1)[1]
        user = await _verify_firebase_token(token)
        if user:
            return user
        raise HTTPException(status_code=401, detail="Invalid Firebase token")

    # 2) API key header fallback
    if API_KEY:
        if request.headers.get("x-api-key") == API_KEY:
            return AuthUser(uid="api-key")
        raise HTTPException(status_code=401, detail="Invalid API key")

    # If auth required but no method configured
    raise HTTPException(status_code=401, detail="Authentication required")


def check_basic_auth(request: Request) -> Optional[AuthUser]:
    """Optional Basic auth using BASIC_AUTH_USER/BASIC_AUTH_PASS envs."""
    user = os.environ.get("BASIC_AUTH_USER")
    pwd = os.environ.get("BASIC_AUTH_PASS")
    if not user or not pwd:
        return None
    authz = request.headers.get("authorization") or request.headers.get("Authorization")
    if not authz or not authz.lower().startswith("basic "):
        return None
    try:
        token = authz.split(" ", 1)[1]
        raw = base64.b64decode(token).decode("utf-8")
        u, p = raw.split(":", 1)
        if u == user and p == pwd:
            return AuthUser(uid=f"basic:{u}")
    except Exception:
        return None
    return None
