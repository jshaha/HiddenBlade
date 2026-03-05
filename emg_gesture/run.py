#!/usr/bin/env python3
"""
Real-Time Inference Entry Point for EMG Gesture Classification.

Runs live gesture classification and actuator control.
"""

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

# Add parent directory to path for imports when running directly
sys.path.insert(0, str(Path(__file__).parent))

import yaml

from data.collector import EMGCollector
from inference.engine import InferenceEngine, MockInferenceEngine
from inference.actuator import ActuatorController, GESTURE_TO_COMMAND


class EMGGestureSystem:
    """
    Complete EMG gesture recognition and actuator control system.

    Coordinates data collection, inference, and actuator control
    in separate threads.
    """

    def __init__(self, config_path: str, mock: bool = False):
        """
        Initialize the gesture recognition system.

        Args:
            config_path: Path to config.yaml file.
            mock: If True, use mock components (no hardware required).
        """
        self.config_path = config_path
        self.mock = mock

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.gesture_classes = self.config['gesture_classes']

        # Components
        self.collector: Optional[EMGCollector] = None
        self.engine: Optional[InferenceEngine] = None
        self.actuator: Optional[ActuatorController] = None

        # Threading
        self._running = False
        self._inference_thread: Optional[threading.Thread] = None
        self._window_queue: Queue = Queue(maxsize=10)

        # Statistics
        self._inference_count = 0
        self._gesture_count = 0
        self._start_time: Optional[float] = None

    def _create_components(self):
        """Create and initialize all system components."""
        print("\n" + "=" * 60)
        print("EMG Gesture Recognition System")
        print("=" * 60)
        print(f"Mode: {'MOCK (no hardware)' if self.mock else 'LIVE'}")
        print(f"Gesture classes: {self.gesture_classes}")
        print("=" * 60)

        # Create collector
        print("\n[System] Initializing data collector...")
        self.collector = EMGCollector(self.config_path, mock=self.mock)
        self.collector.connect()

        # Create inference engine
        print("[System] Initializing inference engine...")
        if self.mock:
            # Check if model checkpoint exists
            checkpoint_path = Path(self.config['paths']['checkpoints_dir']) / 'best_model.pt'
            if checkpoint_path.exists():
                self.engine = InferenceEngine(self.config_path)
            else:
                print("[System] No trained model found, using mock inference")
                self.engine = MockInferenceEngine(self.config_path)
        else:
            self.engine = InferenceEngine(self.config_path)

        # Create actuator controller
        print("[System] Initializing actuator controller...")
        self.actuator = ActuatorController(self.config_path, mock=self.mock)
        self.actuator.connect()

        # Set up gesture callback
        self.engine.set_gesture_callback(self._on_gesture_confirmed)

    def _on_window_ready(self, window):
        """Callback when a new window is ready from collector."""
        try:
            self._window_queue.put_nowait(window)
        except:
            pass  # Drop window if queue is full

    def _on_gesture_confirmed(self, gesture: str, confidence: float):
        """Callback when a stable gesture is confirmed."""
        self._gesture_count += 1

        # Get command byte for display
        command = GESTURE_TO_COMMAND.get(gesture, b'\x00')
        hex_str = command.hex()

        print(f"\n{'='*50}")
        print(f"  GESTURE CONFIRMED: {gesture.upper()}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Command: 0x{hex_str}")
        print(f"{'='*50}\n")

        # Send to actuator
        self.actuator.send_gesture(gesture)

    def _inference_loop(self):
        """Main inference loop running in separate thread."""
        while self._running:
            try:
                # Get window from queue with timeout
                window = self._window_queue.get(timeout=0.1)

                # Run inference
                prediction, confidence, is_new = self.engine.process_window(window)
                self._inference_count += 1

                # Print live prediction
                command = GESTURE_TO_COMMAND.get(prediction, b'\x00')
                status = ">>>" if is_new else "   "
                print(f"{status} Prediction: {prediction:12s} | "
                      f"Confidence: {confidence:.3f} | "
                      f"Cmd: 0x{command.hex()}", end='\r')

            except Empty:
                continue
            except Exception as e:
                print(f"\n[System] Inference error: {e}")

    def start(self):
        """Start the gesture recognition system."""
        self._create_components()

        print("\n[System] Starting system...")
        self._running = True
        self._start_time = time.time()

        # Start inference thread
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True
        )
        self._inference_thread.start()

        # Start data collection (calls _on_window_ready for each window)
        self.collector.start_streaming(callback=self._on_window_ready)

        print("[System] System running. Press Ctrl+C to stop.\n")

    def stop(self):
        """Stop the gesture recognition system gracefully."""
        print("\n\n[System] Shutting down...")
        self._running = False

        # Stop collector
        if self.collector:
            self.collector.stop_streaming()
            self.collector.disconnect()

        # Wait for inference thread
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=2.0)

        # Stop actuator (sends rest command)
        if self.actuator:
            self.actuator.stop()
            self.actuator.disconnect()

        # Print statistics
        if self._start_time:
            elapsed = time.time() - self._start_time
            print("\n" + "=" * 60)
            print("Session Statistics")
            print("=" * 60)
            print(f"Duration: {elapsed:.1f} seconds")
            print(f"Total inferences: {self._inference_count}")
            print(f"Confirmed gestures: {self._gesture_count}")
            if elapsed > 0:
                print(f"Inference rate: {self._inference_count / elapsed:.1f} Hz")
            print("=" * 60)

        print("[System] Shutdown complete")

    def run(self):
        """Run the system until interrupted."""
        # Set up signal handler for graceful shutdown
        def signal_handler(sig, frame):
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            self.start()
            # Keep main thread alive
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main():
    """Main entry point for live inference."""
    parser = argparse.ArgumentParser(
        description='EMG Gesture Recognition - Live Inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mock', action='store_true',
        help='Run in mock mode (no hardware required)'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to model checkpoint (uses default if not specified)'
    )

    args = parser.parse_args()

    # Check config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Run system
    system = EMGGestureSystem(str(config_path), mock=args.mock)
    system.run()


if __name__ == '__main__':
    main()
