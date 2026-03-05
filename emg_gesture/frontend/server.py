#!/usr/bin/env python3
"""
Web Dashboard Server for EMG Gesture Recognition.

Provides a real-time web interface with WebSocket streaming
for EMG signals and gesture predictions.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
from collections import deque

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

from data.collector import EMGCollector
from inference.engine import InferenceEngine, MockInferenceEngine
from inference.actuator import ActuatorController, GESTURE_TO_COMMAND

app = FastAPI(title="EMG Gesture Recognition Dashboard")

# Mount static files
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


class DashboardState:
    """Manages the shared state for the dashboard."""

    def __init__(self):
        self.config_path = str(Path(__file__).parent.parent / "config.yaml")
        self.mock = True
        self.running = False

        self.collector: Optional[EMGCollector] = None
        self.engine: Optional[InferenceEngine] = None
        self.actuator: Optional[ActuatorController] = None

        # Data buffers for visualization
        self.emg_buffer = deque(maxlen=500)  # Last 500 samples
        self.prediction_history = deque(maxlen=100)

        # Current state
        self.current_gesture = "rest"
        self.current_confidence = 0.0
        self.gesture_counts: Dict[str, int] = {}
        self.inference_count = 0
        self.start_time: Optional[float] = None

        # Connected WebSocket clients
        self.clients: list = []

    async def initialize(self, mock: bool = True):
        """Initialize all components."""
        self.mock = mock
        self.running = True
        self.start_time = time.time()

        # Initialize collector
        self.collector = EMGCollector(self.config_path, mock=mock)
        self.collector.connect()

        # Initialize inference engine
        checkpoint_path = Path(self.config_path).parent / "checkpoints" / "best_model.pt"
        if checkpoint_path.exists():
            self.engine = InferenceEngine(self.config_path)
        else:
            self.engine = MockInferenceEngine(self.config_path)

        # Initialize actuator
        self.actuator = ActuatorController(self.config_path, mock=mock)
        self.actuator.connect()

        # Reset counts
        self.gesture_counts = {g: 0 for g in self.engine.gesture_classes}
        self.inference_count = 0

    async def shutdown(self):
        """Shutdown all components."""
        self.running = False

        if self.collector:
            self.collector.disconnect()
        if self.actuator:
            self.actuator.disconnect()

    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        uptime = time.time() - self.start_time if self.start_time else 0

        return {
            "running": self.running,
            "mock_mode": self.mock,
            "current_gesture": self.current_gesture,
            "confidence": self.current_confidence,
            "gesture_counts": self.gesture_counts,
            "inference_count": self.inference_count,
            "uptime": uptime,
            "inference_rate": self.inference_count / uptime if uptime > 0 else 0
        }


# Global state
state = DashboardState()


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the main dashboard page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return FileResponse(html_path)


@app.get("/api/status")
async def get_status():
    """Get current system status."""
    return state.get_status()


@app.post("/api/start")
async def start_system(mock: bool = True):
    """Start the gesture recognition system."""
    if not state.running:
        await state.initialize(mock=mock)
        return {"status": "started", "mock": mock}
    return {"status": "already_running"}


@app.post("/api/stop")
async def stop_system():
    """Stop the gesture recognition system."""
    if state.running:
        await state.shutdown()
        return {"status": "stopped"}
    return {"status": "not_running"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming."""
    await websocket.accept()
    state.clients.append(websocket)

    try:
        # Start streaming if not already running
        if not state.running:
            await state.initialize(mock=True)

        # Stream data
        while state.running:
            # Read EMG sample
            if state.collector and state.collector._mock_generator:
                sample = state.collector._read_sample()
                if sample is not None:
                    state.emg_buffer.append(sample.tolist())

                    # Run inference when we have enough samples
                    if len(state.emg_buffer) >= state.collector.window_size:
                        window = np.array(list(state.emg_buffer)[-state.collector.window_size:])

                        pred, conf, is_new = state.engine.process_window(window)
                        state.current_gesture = pred
                        state.current_confidence = conf
                        state.inference_count += 1

                        if is_new:
                            state.gesture_counts[pred] = state.gesture_counts.get(pred, 0) + 1
                            # Send actuator command
                            state.actuator.send_gesture(pred)

                        # Send update to client
                        data = {
                            "type": "update",
                            "emg": sample.tolist(),
                            "gesture": pred,
                            "confidence": conf,
                            "is_new": is_new,
                            "command": GESTURE_TO_COMMAND.get(pred, b'\x00').hex(),
                            "stats": state.get_status()
                        }
                        await websocket.send_json(data)

            await asyncio.sleep(0.001)  # ~1000Hz

    except WebSocketDisconnect:
        pass
    finally:
        if websocket in state.clients:
            state.clients.remove(websocket)


def main():
    """Run the dashboard server."""
    import argparse

    parser = argparse.ArgumentParser(description='EMG Dashboard Server')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--mock', action='store_true', default=True, help='Use mock mode')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("EMG Gesture Recognition Dashboard")
    print(f"{'='*60}")
    print(f"Open in browser: http://{args.host}:{args.port}")
    print(f"{'='*60}\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
