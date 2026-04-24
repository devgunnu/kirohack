"""
FastAPI server for the MeshRun dashboard.

Provides a websocket endpoint that streams events to the dashboard frontend.
"""

import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.dashboard.events import mock_event_stream

app = FastAPI(title="MeshRun Dashboard")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

connected_clients: list[WebSocket] = []


@app.get("/")
async def serve_dashboard():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
async def get_status():
    # TODO: replace with real coordinator call
    return {
        "active_nodes": 4,
        "total_tokens_processed": 18420,
        "total_cost_saved_usd": 12.43,
        "total_co2_avoided_g": 4.92
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        async for event in mock_event_stream():
            await websocket.send_text(json.dumps(event))
    except WebSocketDisconnect:
        connected_clients.remove(websocket)


def start(host: str = "127.0.0.1", port: int = 7654):
    uvicorn.run(app, host=host, port=port, log_level="error")
