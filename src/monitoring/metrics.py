"""
Monitoring and metrics collection for the AI Assistant System.
Provides comprehensive observability and performance tracking.
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json

from src.logging.logger import logger


class MetricsCollector:
    """Collects and manages system metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.start_time = time.time()
        self.lock = threading.Lock()

        # System metrics
        self.system_metrics = {
            "cpu_usage": deque(maxlen=100),
            "memory_usage": deque(maxlen=100),
            "disk_usage": deque(maxlen=100),
            "network_io": deque(maxlen=100),
        }

        # Application metrics
        self.app_metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "avg_response_time": 0.0,
            "active_sessions": 0,
            "tool_executions": defaultdict(int),
            "api_calls": defaultdict(int),
            "error_count": defaultdict(int),
        }

        # Start background monitoring
        self.monitoring_thread = threading.Thread(
            target=self._monitor_system, daemon=True
        )
        self.monitoring_thread.start()

    def record_request(
        self, duration: float, success: bool, error_type: Optional[str] = None
    ):
        """Record a request metric."""
        with self.lock:
            self.app_metrics["requests_total"] += 1

            if success:
                self.app_metrics["requests_success"] += 1
            else:
                self.app_metrics["requests_failed"] += 1
                if error_type:
                    self.app_metrics["error_count"][error_type] += 1

            # Update average response time
            total_requests = self.app_metrics["requests_total"]
            current_avg = self.app_metrics["avg_response_time"]
            self.app_metrics["avg_response_time"] = (
                current_avg * (total_requests - 1) + duration
            ) / total_requests

            # Record in histogram
            self.histograms["request_duration"].append(duration)

    def record_tool_execution(self, tool_name: str, duration: float, success: bool):
        """Record tool execution metrics."""
        with self.lock:
            self.app_metrics["tool_executions"][tool_name] += 1

            if success:
                self.histograms[f"tool_{tool_name}_success_duration"].append(duration)
            else:
                self.histograms[f"tool_{tool_name}_error_duration"].append(duration)

    def record_api_call(self, api_name: str, duration: float, success: bool):
        """Record API call metrics."""
        with self.lock:
            self.app_metrics["api_calls"][api_name] += 1

            if success:
                self.histograms[f"api_{api_name}_success_duration"].append(duration)
            else:
                self.histograms[f"api_{api_name}_error_duration"].append(duration)

    def increment_counter(self, name: str, value: int = 1):
        """Increment a counter metric."""
        with self.lock:
            self.counters[name] += value

    def set_gauge(self, name: str, value: float):
        """Set a gauge metric."""
        with self.lock:
            self.gauges[name] = value

    def record_histogram(self, name: str, value: float):
        """Record a histogram value."""
        with self.lock:
            self.histograms[name].append(value)

    def _monitor_system(self):
        """Background system monitoring."""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.system_metrics["cpu_usage"].append(
                    {"timestamp": time.time(), "value": cpu_percent}
                )

                # Memory usage
                memory = psutil.virtual_memory()
                self.system_metrics["memory_usage"].append(
                    {
                        "timestamp": time.time(),
                        "value": memory.percent,
                        "available": memory.available,
                        "used": memory.used,
                    }
                )

                # Disk usage
                disk = psutil.disk_usage("/")
                disk_percent = (disk.used / disk.total) * 100
                self.system_metrics["disk_usage"].append(
                    {
                        "timestamp": time.time(),
                        "value": disk_percent,
                        "free": disk.free,
                        "used": disk.used,
                        "total": disk.total,
                    }
                )

                # Network I/O
                network = psutil.net_io_counters()
                self.system_metrics["network_io"].append(
                    {
                        "timestamp": time.time(),
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv,
                        "packets_sent": network.packets_sent,
                        "packets_recv": network.packets_recv,
                    }
                )

                # Update gauges
                self.set_gauge("system_cpu_percent", cpu_percent)
                self.set_gauge("system_memory_percent", memory.percent)
                self.set_gauge("system_disk_percent", disk_percent)
                self.set_gauge("system_uptime", time.time() - self.start_time)

                time.sleep(5)  # Monitor every 5 seconds

            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(10)  # Wait longer on error

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        with self.lock:
            uptime = time.time() - self.start_time

            # Calculate success rate
            total_requests = self.app_metrics["requests_total"]
            success_rate = (
                (self.app_metrics["requests_success"] / total_requests * 100)
                if total_requests > 0
                else 0
            )

            # Calculate percentile for request duration
            durations = self.histograms.get("request_duration", [])
            p50, p95, p99 = self._calculate_percentiles(durations)

            return {
                "system": {
                    "uptime_seconds": uptime,
                    "uptime_human": str(timedelta(seconds=int(uptime))),
                    "cpu_percent": self.gauges.get("system_cpu_percent", 0),
                    "memory_percent": self.gauges.get("system_memory_percent", 0),
                    "disk_percent": self.gauges.get("system_disk_percent", 0),
                },
                "application": {
                    "requests_total": total_requests,
                    "requests_success": self.app_metrics["requests_success"],
                    "requests_failed": self.app_metrics["requests_failed"],
                    "success_rate_percent": success_rate,
                    "avg_response_time": self.app_metrics["avg_response_time"],
                    "response_time_p50": p50,
                    "response_time_p95": p95,
                    "response_time_p99": p99,
                    "active_sessions": self.app_metrics["active_sessions"],
                },
                "tools": dict(self.app_metrics["tool_executions"]),
                "apis": dict(self.app_metrics["api_calls"]),
                "errors": dict(self.app_metrics["error_count"]),
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
            }

    def _calculate_percentiles(
        self, values: List[float], percentiles: List[int] = [50, 95, 99]
    ) -> List[float]:
        """Calculate percentiles for a list of values."""
        if not values:
            return [0.0] * len(percentiles)

        sorted_values = sorted(values)
        results = []

        for p in percentiles:
            index = int((p / 100) * len(sorted_values))
            if index >= len(sorted_values):
                index = len(sorted_values) - 1
            results.append(sorted_values[index])

        return results

    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        summary = self.get_metrics_summary()

        # Define health thresholds
        cpu_threshold = 80.0
        memory_threshold = 85.0
        disk_threshold = 90.0
        success_rate_threshold = 95.0

        health_checks = {
            "cpu": {
                "status": (
                    "healthy"
                    if summary["system"]["cpu_percent"] < cpu_threshold
                    else "warning"
                ),
                "value": summary["system"]["cpu_percent"],
                "threshold": cpu_threshold,
            },
            "memory": {
                "status": (
                    "healthy"
                    if summary["system"]["memory_percent"] < memory_threshold
                    else "warning"
                ),
                "value": summary["system"]["memory_percent"],
                "threshold": memory_threshold,
            },
            "disk": {
                "status": (
                    "healthy"
                    if summary["system"]["disk_percent"] < disk_threshold
                    else "warning"
                ),
                "value": summary["system"]["disk_percent"],
                "threshold": disk_threshold,
            },
            "success_rate": {
                "status": (
                    "healthy"
                    if summary["application"]["success_rate_percent"]
                    > success_rate_threshold
                    else "warning"
                ),
                "value": summary["application"]["success_rate_percent"],
                "threshold": success_rate_threshold,
            },
        }

        # Overall health status
        overall_status = "healthy"
        if any(check["status"] == "warning" for check in health_checks.values()):
            overall_status = "warning"

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "checks": health_checks,
            "summary": summary,
        }

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in various formats."""
        summary = self.get_metrics_summary()

        if format.lower() == "json":
            return json.dumps(summary, indent=2)
        elif format.lower() == "prometheus":
            return self._export_prometheus_format(summary)
        else:
            return str(summary)

    def _export_prometheus_format(self, summary: Dict[str, Any]) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # System metrics
        lines.append(f"system_cpu_percent {summary['system']['cpu_percent']}")
        lines.append(f"system_memory_percent {summary['system']['memory_percent']}")
        lines.append(f"system_disk_percent {summary['system']['disk_percent']}")
        lines.append(f"system_uptime_seconds {summary['system']['uptime_seconds']}")

        # Application metrics
        lines.append(f"requests_total {summary['application']['requests_total']}")
        lines.append(f"requests_success {summary['application']['requests_success']}")
        lines.append(f"requests_failed {summary['application']['requests_failed']}")
        lines.append(
            f"avg_response_time_seconds {summary['application']['avg_response_time']}"
        )

        return "\n".join(lines)


# Global metrics collector instance
metrics_collector = MetricsCollector()


class MetricsMiddleware:
    """Middleware for automatic metrics collection."""

    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize metrics middleware."""
        self.metrics = metrics_collector

    def record_request(self, func):
        """Decorator to record request metrics."""

        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_type = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_type = type(e).__name__
                raise e
            finally:
                duration = time.time() - start_time
                self.metrics.record_request(duration, success, error_type)

        return wrapper

    def record_tool_execution(self, tool_name: str):
        """Decorator to record tool execution metrics."""

        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                success = True

                try:
                    result = func(*args, **kwargs)
                    if isinstance(result, dict) and not result.get("success", True):
                        success = False
                    return result
                except Exception as e:
                    success = False
                    raise e
                finally:
                    duration = time.time() - start_time
                    self.metrics.record_tool_execution(tool_name, duration, success)

            return wrapper

        return decorator


# Convenience functions
def record_request_metric(
    duration: float, success: bool, error_type: Optional[str] = None
):
    """Record a request metric."""
    metrics_collector.record_request(duration, success, error_type)


def record_tool_metric(tool_name: str, duration: float, success: bool):
    """Record a tool execution metric."""
    metrics_collector.record_tool_execution(tool_name, duration, success)


def record_api_metric(api_name: str, duration: float, success: bool):
    """Record an API call metric."""
    metrics_collector.record_api_call(api_name, duration, success)


def get_health_status() -> Dict[str, Any]:
    """Get current health status."""
    return metrics_collector.get_health_status()


def get_metrics_summary() -> Dict[str, Any]:
    """Get metrics summary."""
    return metrics_collector.get_metrics_summary()
