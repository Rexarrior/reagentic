"""
FastAPI server for Protocol Viewer.
"""

from __future__ import annotations

import argparse
import asyncio
import webbrowser
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from reagentic.protocol import SQLiteProtocolStorage, JSONLinesProtocolStorage
from reagentic.protocol.storage.base import ProtocolStorage

from .api import router as api_router, set_storage


def create_storage(db_path: str) -> ProtocolStorage:
    """Create appropriate storage based on file extension."""
    path = Path(db_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    if path.suffix == ".db":
        return SQLiteProtocolStorage(str(path))
    elif path.suffix in (".jsonl", ".json"):
        return JSONLinesProtocolStorage(str(path))
    else:
        # Try SQLite by default
        return SQLiteProtocolStorage(str(path))


def create_app(db_path: str) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Protocol Viewer",
        description="Web-based visualization for agent protocol logs",
        version="1.0.0",
    )
    
    # Add CORS middleware for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize storage
    storage = create_storage(db_path)
    set_storage(storage)
    
    # Include API router
    app.include_router(api_router)
    
    # Serve static files (Vue build)
    frontend_dist = Path(__file__).parent / "frontend" / "dist"
    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")
        
        @app.get("/")
        async def serve_index():
            return FileResponse(str(frontend_dist / "index.html"))
        
        @app.get("/{path:path}")
        async def serve_spa(path: str):
            # Try to serve the file, otherwise return index.html (SPA routing)
            file_path = frontend_dist / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(str(file_path))
            return FileResponse(str(frontend_dist / "index.html"))
    else:
        @app.get("/")
        async def no_frontend():
            return {
                "message": "Frontend not built. Run: cd reagentic/utilities/protocol_viewer/frontend && npm install && npm run build",
                "api_docs": "/docs",
            }
    
    return app


def run_server(
    db_path: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = True,
) -> None:
    """Run the server."""
    app = create_app(db_path)
    
    if open_browser:
        # Open browser after a short delay
        import threading
        def open_browser_delayed():
            import time
            time.sleep(1)
            webbrowser.open(f"http://{host}:{port}")
        threading.Thread(target=open_browser_delayed, daemon=True).start()
    
    print(f"Starting Protocol Viewer at http://{host}:{port}")
    print(f"API docs available at http://{host}:{port}/docs")
    print(f"Reading protocols from: {db_path}")
    
    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Protocol Viewer - Web-based visualization for agent protocol logs"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="protocol.db",
        help="Path to protocol database file (.db or .jsonl)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )
    
    args = parser.parse_args()
    
    run_server(
        db_path=args.db,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
