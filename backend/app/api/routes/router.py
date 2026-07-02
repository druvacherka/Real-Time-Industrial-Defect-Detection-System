"""
routes/router.py — Central API Router
=======================================
Aggregates all endpoint routers into a single router that gets
mounted onto the FastAPI application instance.
"""

from fastapi import APIRouter

from app.api.endpoints import root, health

# Create the top-level API router
api_router = APIRouter()

# Register individual endpoint routers
api_router.include_router(root.router)
api_router.include_router(health.router)
