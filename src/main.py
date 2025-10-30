#!/usr/bin/env python3
"""
GMCS Universal Platform - Main Entry Point

This module provides the CLI interface for launching the GMCS server with
optional audio capture, customizable host/port, and configuration options.

Usage:
    python -m src.main
    python -m src.main --host 0.0.0.0 --port 8080
    python -m src.main --no-audio --debug
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import uvicorn

# ASCII art banner
BANNER = r"""
  ██████╗ ███╗   ███╗ ██████╗███████╗
 ██╔════╝ ████╗ ████║██╔════╝██╔════╝
 ██║  ███╗██╔████╔██║██║     ███████╗
 ██║   ██║██║╚██╔╝██║██║     ╚════██║
 ╚██████╔╝██║ ╚═╝ ██║╚██████╗███████║
  ╚═════╝ ╚═╝     ╚═╝ ╚═════╝╚══════╝
  
Generalized Modular Control System v2.0
Chaotic-Neuro Computational Platform
"""


def setup_logging(debug: bool = False) -> None:
    """Configure structured logging with color-coded severity."""
    
    level = logging.DEBUG if debug else logging.INFO
    
    # Color codes for terminal output
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }
    
    class ColoredFormatter(logging.Formatter):
        """Custom formatter with color-coded severity levels."""
        
        def format(self, record):
            levelname = record.levelname
            if levelname in COLORS:
                record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
            return super().format(record)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColoredFormatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    
    # Quiet down noisy libraries
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.error').setLevel(logging.INFO)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description='GMCS Universal Platform - Chaotic-Neuro Computational Substrate',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                              # Start with defaults
  python -m src.main --host 0.0.0.0 --port 8080   # Custom host/port
  python -m src.main --no-audio                   # Disable audio capture
  python -m src.main --debug                      # Enable debug logging
  
For more information, visit: https://github.com/gavriel-tech/Chaotic-Neuro-Computational-Substrate
        """
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to bind the server to (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind the server to (default: 8000)'
    )
    
    parser.add_argument(
        '--no-audio',
        action='store_true',
        help='Disable audio capture (for headless/server environments)'
    )
    
    parser.add_argument(
        '--audio-device',
        type=int,
        default=None,
        help='Audio input device index (use --list-devices to see options)'
    )
    
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available audio devices and exit'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload for development (WARNING: breaks audio thread)'
    )
    
    return parser.parse_args()


def list_audio_devices() -> None:
    """List available audio input devices."""
    
    try:
        import sounddevice as sd
        
        print("\n=== Available Audio Devices ===\n")
        devices = sd.query_devices()
        
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"[{idx}] {device['name']}")
                print(f"    Channels: {device['max_input_channels']} in")
                print(f"    Sample Rate: {device['default_samplerate']} Hz")
                print()
        
        print("Use --audio-device <index> to select a device\n")
        
    except ImportError:
        print("ERROR: sounddevice not installed. Install with: pip install sounddevice")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to query audio devices: {e}")
        sys.exit(1)


def start_audio_capture(device: int | None, logger: logging.Logger) -> None:
    """Start audio capture thread if audio is enabled."""
    
    try:
        from src.audio.audio_thread import start_audio_capture as start_capture
        from src.api.server import audio_params
        
        logger.info("Starting audio capture thread...")
        start_capture(audio_params, device=device)
        logger.info(f"Audio capture initialized (device: {device if device is not None else 'default'})")
        
    except ImportError as e:
        logger.warning(f"Audio dependencies not available: {e}")
        logger.warning("Audio features will be disabled. Install with: pip install sounddevice librosa")
    except Exception as e:
        logger.error(f"Failed to start audio capture: {e}")
        logger.warning("Continuing without audio...")


def print_startup_info(host: str, port: int, audio_enabled: bool, logger: logging.Logger) -> None:
    """Print startup information and instructions."""
    
    try:
        print(BANNER)
    except UnicodeEncodeError:
        # Fallback for Windows terminals without UTF-8 support
        print("\n=== GMCS - Generalized Modular Control System v2.0 ===")
        print("Chaotic-Neuro Computational Platform\n")
    
    print(f"Server starting on http://{host}:{port}")
    print(f"Audio capture: {'enabled' if audio_enabled else 'disabled'}")
    print()
    print("Quick Start:")
    print(f"  API docs: http://{host}:{port}/docs")
    print(f"  Health check: http://{host}:{port}/health")
    print(f"  Frontend: http://localhost:3000 (run separately)")
    print()
    print("Press CTRL+C to stop")
    print("=" * 60)
    print()
    
    logger.info("GMCS server initialized successfully")


def main() -> int:
    """Main entry point."""
    
    args = parse_arguments()
    
    # Handle --list-devices flag
    if args.list_devices:
        list_audio_devices()
        return 0
    
    # Setup logging
    setup_logging(debug=args.debug)
    logger = logging.getLogger(__name__)
    
    # Start audio capture if enabled
    audio_enabled = not args.no_audio
    if audio_enabled:
        start_audio_capture(args.audio_device, logger)
    
    # Print startup info
    print_startup_info(args.host, args.port, audio_enabled, logger)
    
    # Configure uvicorn
    config = uvicorn.Config(
        "src.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info" if not args.debug else "debug",
        access_log=False,  # We handle our own logging
        ws_ping_interval=20.0,
        ws_ping_timeout=20.0,
        timeout_keep_alive=30,
    )
    
    server = uvicorn.Server(config)
    
    try:
        server.run()
        return 0
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        return 0
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


