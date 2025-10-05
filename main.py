#!/usr/bin/env python3
"""
Main entry point for the AI Assistant System.
Provides command-line interface and system initialization.
"""

import argparse
import sys
import traceback
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.settings import settings
from src.logging.logger import logger
from src.core.ai_system import AISystem


def main():
    """Main entry point for the AI Assistant System."""
    parser = argparse.ArgumentParser(
        description="AI Assistant System - A modular, self-improving AI agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run in text mode
  python main.py --voice            # Run in voice mode
  python main.py --debug            # Run with debug logging
  python main.py --config custom.env # Use custom config file
        """,
    )

    parser.add_argument(
        "--voice", action="store_true", help="Enable voice mode for speech input/output"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with verbose logging"
    )

    parser.add_argument(
        "--config", type=str, help="Path to custom .env configuration file"
    )

    parser.add_argument(
        "--version", action="version", version="AI Assistant System v1.0.0"
    )

    args = parser.parse_args()

    # Load custom config if specified
    if args.config:
        from config.settings import Settings

        global settings
        settings = Settings(args.config)

    # Enable debug mode if requested
    if args.debug:
        settings.DEBUG_MODE = True
        settings.VERBOSE_LOGGING = True

    # Validate API keys
    if not settings.validate_api_keys():
        logger.error(
            "No valid API keys found. Please set GEMINI_API_KEY or GEMINI_API_KEY_SECONDARY in your .env file."
        )
        logger.info("Copy .env.template to .env and fill in your API keys.")
        sys.exit(1)

    # Display startup banner
    logger.banner("AI Assistant System")
    logger.info(f"Base directory: {settings.BASE_DIR}")
    logger.info(f"Voice mode: {'Enabled' if args.voice else 'Disabled'}")
    logger.info(f"Debug mode: {'Enabled' if settings.DEBUG_MODE else 'Disabled'}")
    logger.info(f"API key: {'Primary' if settings.GEMINI_API_KEY else 'Secondary'}")

    try:
        # Initialize and run the AI system
        ai_system = AISystem(voice_mode=args.voice)
        ai_system.run()

    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user. Goodbye!")

    except Exception as e:
        logger.critical(f"Critical error: {e}")
        logger.exception(f"Critical error details: {e}")
        if settings.DEBUG_MODE:
            traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup
        try:
            ai_system.shutdown()
        except:
            pass


if __name__ == "__main__":
    main()
