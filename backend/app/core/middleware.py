"""
API Key Authentication Middleware
===================================
Real-Time Industrial Defect Detection System

Secures prediction endpoints by validating request credentials against the 
configured API_KEY. Logs authentication details and handles unauthorized access.

Author: prajwaledu802-coder
Date: 2026-07-14
"""

import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.config import settings

logger = logging.getLogger("defect_detection.auth_middleware")


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate API Keys for prediction requests.
    Supports reading from 'X-API-Key' header or 'Authorization' Bearer token.
    """

    def __init__(self, app: ASGIApp, api_key_header: str = "X-API-Key"):
        super().__init__(app)
        self.api_key_header = api_key_header

    async def dispatch(self, request: Request, call_next):
        # We only secure prediction endpoints
        if request.url.path.startswith("/predict"):
            client_ip = request.client.host if request.client else "unknown"
            
            # 1. Try to read from X-API-Key header
            api_key = request.headers.get(self.api_key_header)
            
            # 2. Alternatively, try to read from Authorization Header (Bearer token format)
            if not api_key:
                auth_header = request.headers.get("Authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    api_key = auth_header.split(" ", 1)[1]

            # Validation
            expected_key = settings.API_KEY
            
            if not api_key:
                logger.warning(
                    f"[Auth] Unauthorized access attempt from IP: {client_ip} to {request.url.path} - Reason: Missing API Key"
                )
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "unauthorized",
                        "message": "Authentication credentials are missing. Please provide a valid API Key.",
                        "detail": f"Please provide the API key using the header '{self.api_key_header}' or as 'Authorization: Bearer <API_KEY>'."
                    }
                )

            if api_key != expected_key:
                logger.warning(
                    f"[Auth] Unauthorized access attempt from IP: {client_ip} to {request.url.path} - Reason: Invalid API Key"
                )
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "forbidden",
                        "message": "Authentication failed. The provided API Key is invalid.",
                        "detail": "Verify that your API Key is correct and has the required permissions."
                    }
                )

            # Access granted
            logger.info(f"[Auth] Authorized access granted to IP: {client_ip} for {request.url.path}")

        return await call_next(request)
