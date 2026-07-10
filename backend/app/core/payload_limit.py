"""
Middleware to verify request payload size before endpoints execution.
Author: prajwaledu802-coder
"""
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response, status

class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_size_bytes: int):
        super().__init__(app)
        self.max_size_bytes = max_size_bytes

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_size_bytes:
                return Response(
                    content="Request Entity Too Large",
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                )
        return await call_next(request)
