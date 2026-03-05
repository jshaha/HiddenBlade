"""
EMG Data Collection Module.

Handles serial stream ingestion, windowing, and recording of EMG signals.
Supports both real hardware and mock signal generation for testing.
"""

import time
import struct
import threading
from pathlib import Path
from typing import Optional, Callable, Generator
from collections import deque

import numpy as np
import serial
import yaml


class MockSerialGenerator:
    """
    Generates synthetic EMG signals for testing without hardware.

    Produces Gaussian noise with sinusoidal artifacts to simulate
    real EMG signal characteristics.
    """

    def __init__(self, n_channels: int, sample_rate: int, bytes_per_sample: int = 2):
        """
        Initialize the mock signal generator.

        Args:
            n_channels: Number of EMG channels to simulate.
            sample_rate: Sampling rate in Hz.
            bytes_per_sample: Bytes per sample (default 2 for 16-bit).
        """
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.bytes_per_sample = bytes_per_sample
        self._time = 0.0
        self._running = True

        # Artifact frequencies for each channel (muscle activity simulation)
        self._artifact_freqs = np.random.uniform(20, 100, n_channels)
        self._artifact_phases = np.random.uniform(0, 2 * np.pi, n_channels)

    def read_sample(self) -> bytes:
        """
        Generate a single multichannel EMG sample.

        Returns:
            Bytes representing one sample across all channels.
        """
        # Base Gaussian noise
        noise = np.random.randn(self.n_channels) * 100

        # Add sinusoidal artifacts (simulating muscle activity)
        artifacts = 50 * np.sin(
            2 * np.pi * self._artifact_freqs * self._time + self._artifact_phases
        )

        # Add low-frequency drift
        drift = 20 * np.sin(2 * np.pi * 0.5 * self._time)

        # Combine signals
        signal = noise + artifacts + drift

        # Convert to int16 range
        signal = np.clip(signal, -32768, 32767).astype(np.int16)

        # Update time
        self._time += 1.0 / self.sample_rate

        # Pack as bytes (little-endian int16)
        return struct.pack(f'<{self.n_channels}h', *signal)

    def stop(self):
        """Stop the generator."""
        self._running = False


class EMGCollector:
    """
    Collects EMG data from serial port or mock generator.

    Handles buffering, windowing, and saving recordings for offline training.
    """

    def __init__(self, config_path: str, mock: bool = False):
        """
        Initialize the EMG collector.

        Args:
            config_path: Path to config.yaml file.
            mock: If True, use mock signal generator instead of serial port.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.mock = mock
        self.n_channels = self.config['signal']['n_channels']
        self.sample_rate = self.config['signal']['sample_rate']
        self.bytes_per_sample = self.config['signal']['bytes_per_sample']
        self.window_size_ms = self.config['signal']['window_size_ms']
        self.hop_size_ms = self.config['signal']['hop_size_ms']

        # Calculate window parameters in samples
        self.window_size = int(self.window_size_ms * self.sample_rate / 1000)
        self.hop_size = int(self.hop_size_ms * self.sample_rate / 1000)

        # Packet size: all channels * bytes per sample
        self.packet_size = self.n_channels * self.bytes_per_sample

        # Buffer for sliding window
        self.buffer = deque(maxlen=self.window_size)

        # Serial port or mock generator
        self._serial: Optional[serial.Serial] = None
        self._mock_generator: Optional[MockSerialGenerator] = None

        # Threading controls
        self._running = False
        self._collection_thread: Optional[threading.Thread] = None

        # Recordings directory
        self.recordings_dir = Path(self.config['paths']['recordings_dir'])
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

    def connect(self):
        """
        Connect to EMG data source (serial port or mock generator).

        Raises:
            serial.SerialException: If serial port connection fails.
        """
        if self.mock:
            self._mock_generator = MockSerialGenerator(
                self.n_channels,
                self.sample_rate,
                self.bytes_per_sample
            )
            print(f"[Collector] Mock generator initialized: {self.n_channels} channels @ {self.sample_rate}Hz")
        else:
            port = self.config['serial']['emg_port']
            baud = self.config['serial']['emg_baud']
            self._serial = serial.Serial(port, baud, timeout=1.0)
            print(f"[Collector] Connected to {port} @ {baud} baud")

    def disconnect(self):
        """Disconnect from EMG data source."""
        self._running = False

        if self._collection_thread and self._collection_thread.is_alive():
            self._collection_thread.join(timeout=2.0)

        if self._serial and self._serial.is_open:
            self._serial.close()
            print("[Collector] Serial port closed")

        if self._mock_generator:
            self._mock_generator.stop()
            print("[Collector] Mock generator stopped")

    def _read_sample(self) -> Optional[np.ndarray]:
        """
        Read a single multichannel sample.

        Returns:
            Numpy array of shape (n_channels,) or None if read fails.
        """
        try:
            if self.mock:
                data = self._mock_generator.read_sample()
                # Simulate real-time by sleeping
                time.sleep(1.0 / self.sample_rate)
            else:
                data = self._serial.read(self.packet_size)
                if len(data) != self.packet_size:
                    return None

            # Unpack bytes to int16 array
            samples = struct.unpack(f'<{self.n_channels}h', data)
            return np.array(samples, dtype=np.float32)

        except Exception as e:
            print(f"[Collector] Read error: {e}")
            return None

    def _collection_loop(self, callback: Optional[Callable] = None):
        """
        Main collection loop that reads samples and manages the buffer.

        Args:
            callback: Optional callback function called on each new window.
        """
        samples_since_hop = 0

        while self._running:
            sample = self._read_sample()
            if sample is not None:
                self.buffer.append(sample)
                samples_since_hop += 1

                # Check if we should emit a window
                if len(self.buffer) >= self.window_size and samples_since_hop >= self.hop_size:
                    samples_since_hop = 0
                    if callback:
                        window = np.array(list(self.buffer))
                        callback(window)

    def start_streaming(self, callback: Optional[Callable] = None):
        """
        Start continuous data collection in a background thread.

        Args:
            callback: Function called with each new window (numpy array).
        """
        if self._running:
            print("[Collector] Already streaming")
            return

        self._running = True
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(callback,),
            daemon=True
        )
        self._collection_thread.start()
        print("[Collector] Streaming started")

    def stop_streaming(self):
        """Stop continuous data collection."""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=2.0)
        print("[Collector] Streaming stopped")

    def get_current_window(self) -> Optional[np.ndarray]:
        """
        Get the current window from the buffer.

        Returns:
            Numpy array of shape (window_size, n_channels) or None if buffer not full.
        """
        if len(self.buffer) < self.window_size:
            return None
        return np.array(list(self.buffer))

    def stream_windows(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields windows as they become available.

        Yields:
            Numpy arrays of shape (window_size, n_channels).
        """
        samples_since_hop = 0

        while self._running:
            sample = self._read_sample()
            if sample is not None:
                self.buffer.append(sample)
                samples_since_hop += 1

                if len(self.buffer) >= self.window_size and samples_since_hop >= self.hop_size:
                    samples_since_hop = 0
                    yield np.array(list(self.buffer))

    def record_session(self, label: str, duration: float, output_path: Optional[str] = None) -> str:
        """
        Record a labeled EMG session for training data collection.

        Args:
            label: Gesture label for this recording (e.g., 'closed_hand').
            duration: Recording duration in seconds.
            output_path: Optional custom output path; auto-generated if None.

        Returns:
            Path to the saved .npz file.
        """
        if output_path is None:
            timestamp = int(time.time())
            output_path = self.recordings_dir / f"{label}_{timestamp}.npz"
        else:
            output_path = Path(output_path)

        print(f"[Collector] Recording '{label}' for {duration}s...")
        print("[Collector] Starting in 3...")
        time.sleep(1)
        print("[Collector] 2...")
        time.sleep(1)
        print("[Collector] 1...")
        time.sleep(1)
        print("[Collector] GO!")

        # Collect samples
        total_samples = int(duration * self.sample_rate)
        samples = []

        start_time = time.time()
        while len(samples) < total_samples and (time.time() - start_time) < duration + 5:
            sample = self._read_sample()
            if sample is not None:
                samples.append(sample)

            # Progress indicator
            if len(samples) % self.sample_rate == 0:
                elapsed = len(samples) / self.sample_rate
                print(f"[Collector] Recorded {elapsed:.1f}s / {duration}s")

        samples = np.array(samples)

        # Save to disk
        np.savez(
            output_path,
            data=samples,
            label=label,
            sample_rate=self.sample_rate,
            n_channels=self.n_channels,
            timestamp=time.time()
        )

        print(f"[Collector] Saved {len(samples)} samples to {output_path}")
        return str(output_path)


def main():
    """CLI for recording EMG data."""
    import argparse

    parser = argparse.ArgumentParser(description='EMG Data Collector')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock signal generator')
    parser.add_argument('--record', type=str, metavar='LABEL',
                        help='Record a session with given label')
    parser.add_argument('--duration', type=float, default=10.0,
                        help='Recording duration in seconds')
    parser.add_argument('--stream', action='store_true',
                        help='Stream data and print windows')

    args = parser.parse_args()

    collector = EMGCollector(args.config, mock=args.mock)
    collector.connect()

    try:
        if args.record:
            collector.record_session(args.record, args.duration)
        elif args.stream:
            collector._running = True
            print("[Collector] Streaming windows (Ctrl+C to stop)...")
            for i, window in enumerate(collector.stream_windows()):
                print(f"Window {i}: shape={window.shape}, "
                      f"mean={window.mean():.2f}, std={window.std():.2f}")
                if i >= 10:  # Stop after 10 windows for demo
                    break
    except KeyboardInterrupt:
        print("\n[Collector] Interrupted")
    finally:
        collector.disconnect()


if __name__ == '__main__':
    main()
