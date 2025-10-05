"""
Health check and monitoring endpoints for the AI Assistant System.
Provides REST API endpoints for system monitoring and observability.
"""

import json
import time
from typing import Dict, Any
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

from src.monitoring import get_health_status, get_metrics_summary, metrics_collector
from src.logging.logger import logger


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP request handler for health check endpoints."""

    def __init__(self, ai_system, *args, **kwargs):
        """Initialize health handler with AI system reference."""
        self.ai_system = ai_system
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        try:
            if self.path == "/health":
                self._handle_health()
            elif self.path == "/ready":
                self._handle_ready()
            elif self.path == "/metrics":
                self._handle_metrics()
            elif self.path == "/status":
                self._handle_status()
            else:
                self._send_response(404, {"error": "Not found"})
        except Exception as e:
            logger.error(f"Error handling request {self.path}: {e}")
            self._send_response(500, {"error": "Internal server error"})

    def _handle_health(self):
        """Handle health check endpoint."""
        health_status = get_health_status()

        # Determine HTTP status code
        if health_status["status"] == "healthy":
            status_code = 200
        elif health_status["status"] == "warning":
            status_code = 200  # Still operational
        else:
            status_code = 503  # Service unavailable

        self._send_response(status_code, health_status)

    def _handle_ready(self):
        """Handle readiness check endpoint."""
        # Check if system is ready to accept requests
        ready_checks = {
            "ai_system_active": self.ai_system.active if self.ai_system else False,
            "context_manager_ready": (
                hasattr(self.ai_system, "context_manager") if self.ai_system else False
            ),
            "tool_manager_ready": (
                hasattr(self.ai_system, "tool_manager") if self.ai_system else False
            ),
            "gemini_client_ready": (
                hasattr(self.ai_system, "gemini_client") if self.ai_system else False
            ),
        }

        is_ready = all(ready_checks.values())

        response = {"ready": is_ready, "checks": ready_checks, "timestamp": time.time()}

        status_code = 200 if is_ready else 503
        self._send_response(status_code, response)

    def _handle_metrics(self):
        """Handle metrics endpoint."""
        metrics = get_metrics_summary()
        self._send_response(200, metrics)

    def _handle_status(self):
        """Handle general status endpoint."""
        status = {
            "system": "ai-assistant-system",
            "version": "1.0.0",
            "uptime": metrics_collector.gauges.get("system_uptime", 0),
            "status": "running",
            "timestamp": time.time(),
        }

        self._send_response(200, status)

    def _send_response(self, status_code: int, data: Dict[str, Any]):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        response_json = json.dumps(data, indent=2)
        self.wfile.write(response_json.encode("utf-8"))

    def log_message(self, format, *args):
        """Override to use our logger instead of default."""
        logger.info(f"HTTP {format % args}")


class HealthServer:
    """Health check server for monitoring endpoints."""

    def __init__(self, ai_system, host: str = "localhost", port: int = 8000):
        """Initialize health server."""
        self.ai_system = ai_system
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False

    def start(self):
        """Start the health server."""
        try:
            # Create handler with AI system reference
            handler = lambda *args, **kwargs: HealthHandler(
                self.ai_system, *args, **kwargs
            )

            self.server = HTTPServer((self.host, self.port), handler)
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            self.running = True

            logger.info(f"Health server started on http://{self.host}:{self.port}")
            logger.info(f"Available endpoints:")
            logger.info(f"  - GET /health - Health check")
            logger.info(f"  - GET /ready - Readiness check")
            logger.info(f"  - GET /metrics - System metrics")
            logger.info(f"  - GET /status - General status")

        except Exception as e:
            logger.error(f"Failed to start health server: {e}")

    def stop(self):
        """Stop the health server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.running = False
            logger.info("Health server stopped")

    def _run_server(self):
        """Run the HTTP server."""
        try:
            self.server.serve_forever()
        except Exception as e:
            logger.error(f"Health server error: {e}")

    def is_running(self) -> bool:
        """Check if server is running."""
        return self.running


# Convenience functions for external monitoring
def start_health_server(
    ai_system, host: str = "localhost", port: int = 8000
) -> HealthServer:
    """Start health check server."""
    server = HealthServer(ai_system, host, port)
    server.start()
    return server


def check_health() -> Dict[str, Any]:
    """Check system health (for external monitoring)."""
    return get_health_status()


def get_system_metrics() -> Dict[str, Any]:
    """Get system metrics (for external monitoring)."""
    return get_metrics_summary()


# Example usage and testing
if __name__ == "__main__":
    # Test the health endpoints
    import requests

    # Start server
    server = HealthServer(None)  # No AI system for testing
    server.start()

    try:
        # Wait for server to start
        time.sleep(1)

        # Test endpoints
        base_url = f"http://{server.host}:{server.port}"

        print("Testing health endpoints...")

        # Test health
        response = requests.get(f"{base_url}/health")
        print(f"Health: {response.status_code} - {response.json()}")

        # Test ready
        response = requests.get(f"{base_url}/ready")
        print(f"Ready: {response.status_code} - {response.json()}")

        # Test metrics
        response = requests.get(f"{base_url}/metrics")
        print(f"Metrics: {response.status_code} - {response.json()}")

        # Test status
        response = requests.get(f"{base_url}/status")
        print(f"Status: {response.status_code} - {response.json()}")

    finally:
        server.stop()
