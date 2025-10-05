"""
API package for the AI Assistant System.
Provides REST API endpoints for external integration and monitoring.
"""

from .health import HealthServer, start_health_server, check_health, get_system_metrics

__all__ = ["HealthServer", "start_health_server", "check_health", "get_system_metrics"]
