#!/usr/bin/env python3
"""
Launch the EMG Gesture Recognition Dashboard.

A beautiful real-time web interface for visualizing EMG signals
and gesture predictions.
"""

import os
import sys
import webbrowser
import threading
import time
from pathlib import Path

# Change to the script's directory for proper relative imports
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Add paths for imports
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir / 'frontend'))

def open_browser(url: str, delay: float = 1.5):
    """Open the dashboard in the default browser after a delay."""
    time.sleep(delay)
    webbrowser.open(url)


def main():
    """Launch the dashboard server."""
    import argparse

    parser = argparse.ArgumentParser(
        description='EMG Gesture Recognition Dashboard',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--host', type=str, default='127.0.0.1',
        help='Host to bind to'
    )
    parser.add_argument(
        '--port', type=int, default=8000,
        help='Port to bind to'
    )
    parser.add_argument(
        '--no-browser', action='store_true',
        help='Do not open browser automatically'
    )

    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     ███████╗███╗   ███╗ ██████╗                              ║
    ║     ██╔════╝████╗ ████║██╔════╝                              ║
    ║     █████╗  ██╔████╔██║██║  ███╗                             ║
    ║     ██╔══╝  ██║╚██╔╝██║██║   ██║                             ║
    ║     ███████╗██║ ╚═╝ ██║╚██████╔╝                             ║
    ║     ╚══════╝╚═╝     ╚═╝ ╚═════╝                              ║
    ║                                                              ║
    ║     GESTURE RECOGNITION DASHBOARD                            ║
    ║     BiLSTM-CNN Neural Interface                              ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║""")
    print(f"    ║     Dashboard URL: {url:<40} ║")
    print("""    ║                                                              ║
    ║     Press Ctrl+C to stop the server                          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Open browser automatically
    if not args.no_browser:
        threading.Thread(
            target=open_browser,
            args=(url,),
            daemon=True
        ).start()

    # Import and run server
    import uvicorn
    from server import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == '__main__':
    main()
