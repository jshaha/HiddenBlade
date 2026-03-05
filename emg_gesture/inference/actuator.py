"""
Actuator Controller for EMG Gesture System.

Maps gesture labels to serial commands for mechanical actuator control.
"""

from typing import Optional

import serial
import yaml


# Gesture to command byte mapping
GESTURE_TO_COMMAND = {
    "closed_hand": b'\x01',
    "open_hand":   b'\x02',
    "pointing":    b'\x03',
    "rest":        b'\x00',
}


class ActuatorController:
    """
    Controls a mechanical actuator via serial communication.

    Maps gesture labels to command bytes and handles serial transmission.
    """

    def __init__(self, config_path: str, mock: bool = False):
        """
        Initialize the actuator controller.

        Args:
            config_path: Path to config.yaml file.
            mock: If True, print commands instead of sending to serial port.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.mock = mock
        self.port = self.config['serial']['actuator_port']
        self.baud = self.config['serial']['actuator_baud']

        self._serial: Optional[serial.Serial] = None
        self._last_command: Optional[bytes] = None
        self._connected = False

    def connect(self) -> bool:
        """
        Connect to the actuator serial port.

        Returns:
            True if connection successful, False otherwise.
        """
        if self.mock:
            print(f"[Actuator] Mock mode - not opening serial port")
            self._connected = True
            return True

        try:
            self._serial = serial.Serial(
                self.port,
                self.baud,
                timeout=1.0,
                write_timeout=1.0
            )
            self._connected = True
            print(f"[Actuator] Connected to {self.port} @ {self.baud} baud")
            return True

        except serial.SerialException as e:
            print(f"[Actuator] Failed to connect: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from the actuator serial port."""
        if self._serial and self._serial.is_open:
            # Send rest command before disconnecting
            self.send_gesture("rest")
            self._serial.close()
            print("[Actuator] Disconnected")

        self._connected = False

    def send_command(self, command: bytes) -> bool:
        """
        Send a raw command byte to the actuator.

        Args:
            command: Command byte(s) to send.

        Returns:
            True if command sent successfully, False otherwise.
        """
        if not self._connected:
            print("[Actuator] Not connected")
            return False

        # Skip if same as last command (avoid redundant sends)
        if command == self._last_command:
            return True

        self._last_command = command

        if self.mock:
            # Print command in hex format
            hex_str = command.hex()
            print(f"[Actuator] Mock send: 0x{hex_str}")
            return True

        try:
            self._serial.write(command)
            self._serial.flush()
            return True

        except serial.SerialException as e:
            print(f"[Actuator] Write error: {e}")
            return False

        except serial.SerialTimeoutException:
            print("[Actuator] Write timeout")
            return False

    def send_gesture(self, gesture: str) -> bool:
        """
        Send the command corresponding to a gesture label.

        Args:
            gesture: Gesture label (e.g., 'closed_hand', 'open_hand').

        Returns:
            True if command sent successfully, False otherwise.
        """
        if gesture not in GESTURE_TO_COMMAND:
            print(f"[Actuator] Unknown gesture: {gesture}")
            # Default to rest for unknown gestures
            gesture = "rest"

        command = GESTURE_TO_COMMAND[gesture]
        return self.send_command(command)

    def stop(self):
        """Send stop command (rest state)."""
        self.send_gesture("rest")

    @property
    def is_connected(self) -> bool:
        """Check if actuator is connected."""
        return self._connected

    @staticmethod
    def get_command_for_gesture(gesture: str) -> bytes:
        """
        Get the command byte for a gesture without sending.

        Args:
            gesture: Gesture label.

        Returns:
            Command byte for the gesture, or rest command if unknown.
        """
        return GESTURE_TO_COMMAND.get(gesture, GESTURE_TO_COMMAND["rest"])


class MockActuatorController(ActuatorController):
    """
    Mock actuator controller that logs commands without serial communication.

    Useful for testing and debugging the full pipeline.
    """

    def __init__(self, config_path: str, **kwargs):
        """
        Initialize mock actuator controller.

        Args:
            config_path: Path to config.yaml file.
            **kwargs: Additional arguments (ignored).
        """
        super().__init__(config_path, mock=True)

    def connect(self) -> bool:
        """Always succeeds in mock mode."""
        print("[MockActuator] Connected (mock mode)")
        self._connected = True
        return True

    def disconnect(self):
        """No-op in mock mode."""
        print("[MockActuator] Disconnected (mock mode)")
        self._connected = False


def main():
    """Test the actuator controller."""
    import argparse
    import time

    parser = argparse.ArgumentParser(description='Actuator Controller Test')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock mode (no serial port)')

    args = parser.parse_args()

    # Create controller
    controller = ActuatorController(args.config, mock=args.mock)

    if not controller.connect():
        print("Failed to connect to actuator")
        return

    try:
        print("\nTesting gesture commands...")
        gestures = ["rest", "closed_hand", "open_hand", "pointing", "rest"]

        for gesture in gestures:
            print(f"\nSending: {gesture}")
            command = GESTURE_TO_COMMAND[gesture]
            print(f"  Command byte: 0x{command.hex()}")
            controller.send_gesture(gesture)
            time.sleep(1.0)

        print("\nTest complete")

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        controller.disconnect()


if __name__ == '__main__':
    main()
