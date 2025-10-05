"""
Core AI System for the AI Assistant System.
Main orchestrator that handles AI interactions, tool management, and task execution.
"""

import json
import os
import re
import time
import hashlib
from collections import deque
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import requests

# Add parent directory to path for imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config.settings import settings
from src.logging.logger import logger
from src.database.memory import MemoryManager, ToolManager, ExecutionHistory, Tool
from src.tools.base_tools import BaseTools
from src.tools.voice_tools import VoiceTools
from src.core.context_manager import ContextManager
from src.tools.google_search import GoogleSearchTool
from src.core.gemini_client import GeminiClient
from src.monitoring import metrics_collector, record_request_metric, record_tool_metric
from src.api import start_health_server


class AISystem:
    """
    Core AI system that orchestrates all functionality.
    Handles AI interactions, tool management, and task execution.
    """

    def __init__(self, voice_mode: bool = None):
        """Initialize the AI system."""
        self.voice_mode = (
            voice_mode if voice_mode is not None else settings.VOICE_ENABLED
        )
        self.logger = logger

        # Initialize components
        self.memory_manager = MemoryManager()
        self.tool_manager = ToolManager()
        self.execution_history = ExecutionHistory()
        self.context_manager = ContextManager()
        self.google_search = GoogleSearchTool()
        self.gemini_client = GeminiClient()

        # Load system configuration
        self.system_config = self._load_system_config()

        # Initialize tools
        self.base_tools = BaseTools(system=self)
        self.voice_tools = VoiceTools(system=self)

        # System context - completely dynamic
        self.context = {
            "cwd": os.getcwd(),  # Start from actual current working directory
            "os": sys.platform,
            "python_version": sys.version,
            "system_info": {},
            "initial_goal": None,
            "pending_goal": None,
            "api_key_in_use": "primary",
            "execution_history": deque(maxlen=20),
            "conversation_history": deque(maxlen=20),
        }

        self.active = True

        # Register core tools
        self._register_core_tools()
        self._load_tools_from_files()
        self._get_initial_system_details()

        # Start health monitoring server
        self.health_server = None
        if settings.DEBUG_MODE or getattr(settings, "ENABLE_HEALTH_SERVER", True):
            try:
                # Try different ports to avoid conflicts
                for port in [8001, 8002, 8003, 8004, 8005]:
                    try:
                        self.health_server = start_health_server(self, port=port)
                        self.logger.info(
                            f"Health monitoring server started on port {port}"
                        )
                        break
                    except OSError as e:
                        if "Address already in use" in str(e):
                            continue
                        else:
                            raise e
                else:
                    self.logger.warning(
                        "Could not find available port for health server"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to start health server: {e}")

        # Warm up caches with common data
        self._warmup_caches()

        self.logger.success("AI System initialized successfully")

    def _warmup_caches(self):
        """Warm up caches with common data for better performance."""
        try:
            # Check API key status before warmup
            api_status = self.gemini_client.get_api_key_status()
            if (
                api_status.get("primary", {}).get("status") == "rate_limited"
                and api_status.get("secondary", {}).get("status") == "rate_limited"
            ):
                self.logger.warning(
                    "Both API keys are rate limited, skipping cache warmup"
                )
                return

            # Warm up Gemini cache with common prompts (only if API is available)
            common_prompts = [
                "What is the current time?",
                "Help me with a simple task",
                "Explain how to use this system",
            ]

            try:
                self.gemini_client.warmup_cache(common_prompts)
                self.logger.info("Gemini cache warmed up successfully")
            except Exception as e:
                self.logger.warning(f"Gemini cache warmup failed: {e}")

            # Optimize context storage (this doesn't require API calls)
            try:
                self.context_manager.optimize_context_storage()
                self.logger.info("Context storage optimized successfully")
            except Exception as e:
                self.logger.warning(f"Context storage optimization failed: {e}")

        except Exception as e:
            self.logger.warning(f"Failed to warm up caches: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        try:
            return {
                "gemini_cache": self.gemini_client.get_cache_stats(),
                "context_cache": self.context_manager.get_cache_stats(),
                "system_metrics": metrics_collector.get_metrics_summary(),
            }
        except Exception as e:
            self.logger.error(f"Error getting performance stats: {e}")
            return {}

    def optimize_performance(self):
        """Run system performance optimization."""
        try:
            # Optimize Gemini cache
            self.gemini_client.clear_cache()

            # Optimize context storage
            self.context_manager.optimize_context_storage()

            # Run system optimization
            metrics_collector.optimize_memory()

            self.logger.info("System performance optimized")

        except Exception as e:
            self.logger.error(f"Error optimizing performance: {e}")

    def _load_system_config(self) -> Dict[str, Any]:
        """Load system configuration from system.json file."""
        try:
            system_file = Path(__file__).parent.parent.parent / "system.json"
            if system_file.exists():
                with open(system_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                self.logger.warning(
                    "system.json not found, using default configuration"
                )
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load system configuration: {e}")
            return {}

    def _get_system_rules(self) -> str:
        """Generate system rules from configuration."""
        if not self.system_config:
            return "- **NEVER** create files or directories in the project root directory\n- **NEVER** create files or directories in system directories\n- **When user doesn't specify a location**: Create files/directories in the `output/` folder for organization"

        system_dirs = self.system_config.get("system_directories", [])
        protected_files = self.system_config.get("protected_files", [])
        output_dir = self.system_config.get("output_directory", "output/")

        rules = []
        rules.append(
            "- **NEVER** create files or directories in the project root directory (where main.py is located)"
        )

        if system_dirs:
            rules.append(
                f"- **NEVER** create files or directories in system directories: {', '.join(system_dirs)}"
            )

        if protected_files:
            rules.append(
                f"- **NEVER** modify protected files: {', '.join(protected_files)}"
            )

        rules.append(
            "- **When user specifies a location**: Use the exact path the user provides (absolute or relative)"
        )
        rules.append(
            f"- **When user doesn't specify a location**: Create files/directories in the `{output_dir}` folder for organization"
        )
        rules.append("- **Examples of proper usage**:")
        rules.append(f'  - User says "create file.txt" → use `{output_dir}file.txt`')
        rules.append(
            '  - User says "create /home/user/document.pdf" → use `/home/user/document.pdf`'
        )
        rules.append(
            f'  - User says "create data/analysis.json" → use `{output_dir}data/analysis.json`'
        )
        rules.append(
            f'  - User says "create in current directory" → use `{output_dir}` (not project root)'
        )
        rules.append(
            "- **Always ask for clarification** if the intended location is unclear"
        )
        rules.append("- This keeps the system organized and respects user intentions")

        return "\n".join(rules)

    def _create_project_tool(
        self, project_name: str, project_path: str
    ) -> Dict[str, Any]:
        """Tool wrapper for creating a project."""
        try:
            project_id = self.context_manager.create_project(project_name, project_path)
            return {
                "success": True,
                "output": f"Created project '{project_name}' with ID: {project_id}",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to create project: {e}"}

    def _set_active_project_tool(self, project_id: str) -> Dict[str, Any]:
        """Tool wrapper for setting active project."""
        try:
            success = self.context_manager.set_active_project(project_id)
            if success:
                project = self.context_manager.get_active_project()
                return {
                    "success": True,
                    "output": f"Activated project: {project.project_name} at {project.project_path}",
                }
            else:
                return {"success": False, "error": "Project not found"}
        except Exception as e:
            return {"success": False, "error": f"Failed to set active project: {e}"}

    def _get_context_summary_tool(self) -> Dict[str, Any]:
        """Tool wrapper for getting context summary."""
        try:
            summary = self.context_manager.get_context_summary()
            return {"success": True, "output": summary}
        except Exception as e:
            return {"success": False, "error": f"Failed to get context summary: {e}"}

    def _search_context_by_time_tool(
        self, start_time: float = None, end_time: float = None, entry_type: str = None
    ) -> Dict[str, Any]:
        """Tool wrapper for searching context by time range."""
        try:
            results = self.context_manager.search_context_by_time(
                start_time, end_time, entry_type
            )
            if results:
                formatted_results = []
                for entry in results:
                    timestamp = entry.get("timestamp", 0)
                    time_str = (
                        datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                        if timestamp
                        else "Unknown"
                    )
                    formatted_results.append(
                        f"[{time_str}] {entry.get('entry_type', 'unknown')}: {entry.get('content', '')[:200]}..."
                    )
                return {
                    "success": True,
                    "output": f"Found {len(results)} entries:\n"
                    + "\n".join(formatted_results),
                }
            else:
                return {
                    "success": True,
                    "output": "No entries found for the specified time range",
                }
        except Exception as e:
            return {"success": False, "error": f"Failed to search context by time: {e}"}

    def _get_context_by_date_tool(self, date_str: str) -> Dict[str, Any]:
        """Tool wrapper for getting context by date."""
        try:
            results = self.context_manager.get_context_by_date(date_str)
            if results:
                formatted_results = []
                for entry in results:
                    timestamp = entry.get("timestamp", 0)
                    time_str = (
                        datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
                        if timestamp
                        else "Unknown"
                    )
                    formatted_results.append(
                        f"[{time_str}] {entry.get('entry_type', 'unknown')}: {entry.get('content', '')[:200]}..."
                    )
                return {
                    "success": True,
                    "output": f"Found {len(results)} entries for {date_str}:\n"
                    + "\n".join(formatted_results),
                }
            else:
                return {
                    "success": True,
                    "output": f"No entries found for date {date_str}",
                }
        except Exception as e:
            return {"success": False, "error": f"Failed to get context by date: {e}"}

    def _get_context_by_hour_tool(self, hour: int) -> Dict[str, Any]:
        """Tool wrapper for getting context by hour."""
        try:
            results = self.context_manager.get_context_by_hour(hour)
            if results:
                formatted_results = []
                for entry in results:
                    timestamp = entry.get("timestamp", 0)
                    time_str = (
                        datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
                        if timestamp
                        else "Unknown"
                    )
                    formatted_results.append(
                        f"[{time_str}] {entry.get('entry_type', 'unknown')}: {entry.get('content', '')[:200]}..."
                    )
                return {
                    "success": True,
                    "output": f"Found {len(results)} entries for hour {hour:02d}:00:\n"
                    + "\n".join(formatted_results),
                }
            else:
                return {
                    "success": True,
                    "output": f"No entries found for hour {hour:02d}:00",
                }
        except Exception as e:
            return {"success": False, "error": f"Failed to get context by hour: {e}"}

    def _register_core_tools(self):
        """Register core tools with the system."""
        self.logger.info("Registering core tools...")

        core_tools = [
            Tool(
                name="run_shell",
                code="",
                doc="Execute a shell command. Usage: run_shell(command)",
                is_dynamic=False,
                func=self.base_tools.run_shell,
            ),
            Tool(
                name="run_shell_async",
                code="",
                doc="Execute shell commands asynchronously for blocking commands like browsers. Usage: run_shell_async(command, timeout=300)",
                is_dynamic=False,
                func=self.base_tools.run_shell_async,
            ),
            Tool(
                name="interact_with_process",
                code="",
                doc="Interact with a running async process. Usage: interact_with_process(process_id, action='status', data=None)",
                is_dynamic=False,
                func=self.base_tools.interact_with_process,
            ),
            Tool(
                name="create_and_save_tool",
                code="",
                doc="Create and register a new tool. Usage: create_and_save_tool(tool_name, tool_code, doc_string)",
                is_dynamic=False,
                func=self.base_tools.create_and_save_tool,
            ),
            Tool(
                name="install_package",
                code="",
                doc="Install a Python package. Usage: install_package(package_name)",
                is_dynamic=False,
                func=self.base_tools.install_package,
            ),
            Tool(
                name="read_file",
                code="",
                doc="Read file contents. Usage: read_file(file_path)",
                is_dynamic=False,
                func=self.base_tools.read_file,
            ),
            Tool(
                name="write_file",
                code="",
                doc="Write content to file. Usage: write_file(file_path, content)",
                is_dynamic=False,
                func=self.base_tools.write_file,
            ),
            Tool(
                name="list_dir",
                code="",
                doc="List directory contents. Usage: list_dir(directory='.')",
                is_dynamic=False,
                func=self.base_tools.list_dir,
            ),
            Tool(
                name="change_dir",
                code="",
                doc="Change current working directory. Usage: change_dir(directory)",
                is_dynamic=False,
                func=self.base_tools.change_dir,
            ),
            Tool(
                name="complete_task",
                code="",
                doc="Mark task as complete. Usage: complete_task(message)",
                is_dynamic=False,
                func=self.base_tools.complete_task,
            ),
            Tool(
                name="get_system_info",
                code="",
                doc="Get system information. Usage: get_system_info()",
                is_dynamic=False,
                func=self.base_tools.get_system_info,
            ),
            Tool(
                name="google_search",
                code="",
                doc="Search the web using Google. Usage: google_search(query, num_results=10)",
                is_dynamic=False,
                func=self.google_search.search,
            ),
            Tool(
                name="google_search_news",
                code="",
                doc="Search for news articles. Usage: google_search_news(query, num_results=10)",
                is_dynamic=False,
                func=self.google_search.search_news,
            ),
            Tool(
                name="google_search_images",
                code="",
                doc="Search for images. Usage: google_search_images(query, num_results=10)",
                is_dynamic=False,
                func=self.google_search.search_images,
            ),
            Tool(
                name="create_project",
                code="",
                doc="Create a new project context. Usage: create_project(project_name, project_path)",
                is_dynamic=False,
                func=self._create_project_tool,
            ),
            Tool(
                name="set_active_project",
                code="",
                doc="Set active project. Usage: set_active_project(project_id)",
                is_dynamic=False,
                func=self._set_active_project_tool,
            ),
            Tool(
                name="get_context_summary",
                code="",
                doc="Get current context summary. Usage: get_context_summary()",
                is_dynamic=False,
                func=self._get_context_summary_tool,
            ),
            Tool(
                name="search_context_by_time",
                code="",
                doc="Search context entries by time range. Usage: search_context_by_time(start_time=None, end_time=None, entry_type=None)",
                is_dynamic=False,
                func=self._search_context_by_time_tool,
            ),
            Tool(
                name="get_context_by_date",
                code="",
                doc="Get context entries for a specific date. Usage: get_context_by_date(date_str) - format: YYYY-MM-DD",
                is_dynamic=False,
                func=self._get_context_by_date_tool,
            ),
            Tool(
                name="get_context_by_hour",
                code="",
                doc="Get context entries for a specific hour. Usage: get_context_by_hour(hour) - hour: 0-23",
                is_dynamic=False,
                func=self._get_context_by_hour_tool,
            ),
            Tool(
                name="analyze_image",
                code="",
                doc="Analyze an image using Gemini Vision. Usage: analyze_image(image_path, prompt='Describe this image')",
                is_dynamic=False,
                func=self.gemini_client.analyze_image,
            ),
            Tool(
                name="read_screen",
                code="",
                doc="Capture the current screen, analyze it with AI, and clean up the temporary image. Usage: read_screen(prompt='Describe what you see on this screen')",
                is_dynamic=False,
                func=self.base_tools.read_screen,
            ),
            Tool(
                name="click_screen",
                code="",
                doc="Click at specific screen coordinates. Usage: click_screen(x, y, button='left')",
                is_dynamic=False,
                func=self.base_tools.click_screen,
            ),
            Tool(
                name="get_mouse_position",
                code="",
                doc="Get current mouse position. Usage: get_mouse_position()",
                is_dynamic=False,
                func=self.base_tools.get_mouse_position,
            ),
            Tool(
                name="get_active_window",
                code="",
                doc="Get information about the currently active/focused window. Usage: get_active_window()",
                is_dynamic=False,
                func=self.base_tools.get_active_window,
            ),
            Tool(
                name="get_all_windows",
                code="",
                doc="Get list of all visible windows with their details. Usage: get_all_windows()",
                is_dynamic=False,
                func=self.base_tools.get_all_windows,
            ),
            Tool(
                name="focus_window",
                code="",
                doc="Focus/activate a specific window by ID, name, or class. Usage: focus_window(window_identifier)",
                is_dynamic=False,
                func=self.base_tools.focus_window,
            ),
            Tool(
                name="bring_window_to_front",
                code="",
                doc="Dynamically bring any window to the foreground. Usage: bring_window_to_front(window_identifier=None)",
                is_dynamic=False,
                func=self.base_tools.bring_window_to_front,
            ),
            Tool(
                name="check_browser_status",
                code="",
                doc="Check if browser is running and active. Usage: check_browser_status()",
                is_dynamic=False,
                func=self.base_tools.check_browser_status,
            ),
            Tool(
                name="install_system_package",
                code="",
                doc="Install system packages using package manager. Usage: install_system_package(package_name)",
                is_dynamic=False,
                func=self.base_tools.install_system_package,
            ),
            Tool(
                name="check_system_dependency",
                code="",
                doc="Check if system dependency is installed and get installation instructions. Usage: check_system_dependency(dependency_name)",
                is_dynamic=False,
                func=self.base_tools.check_system_dependency,
            ),
            Tool(
                name="analyze_screen_actions",
                code="",
                doc="Analyze screen and provide actionable steps with coordinates and operations. Usage: analyze_screen_actions(task_description)",
                is_dynamic=False,
                func=self.base_tools.analyze_screen_actions,
            ),
            Tool(
                name="scroll_screen",
                code="",
                doc="Scroll at specific screen coordinates. Usage: scroll_screen(x, y, direction='up', amount=3)",
                is_dynamic=False,
                func=self.base_tools.scroll_screen,
            ),
            Tool(
                name="move_mouse",
                code="",
                doc="Move mouse to specific coordinates. Usage: move_mouse(x, y)",
                is_dynamic=False,
                func=self.base_tools.move_mouse,
            ),
            Tool(
                name="drag_mouse",
                code="",
                doc="Drag mouse from one point to another. Usage: drag_mouse(x1, y1, x2, y2, duration=1.0)",
                is_dynamic=False,
                func=self.base_tools.drag_mouse,
            ),
            Tool(
                name="type_text",
                code="",
                doc="Type text at specific coordinates or current position. Usage: type_text(text, x=None, y=None)",
                is_dynamic=False,
                func=self.base_tools.type_text,
            ),
            Tool(
                name="press_key",
                code="",
                doc="Press key combination. Usage: press_key(key_combination)",
                is_dynamic=False,
                func=self.base_tools.press_key,
            ),
            Tool(
                name="generate_structured_output",
                code="",
                doc="Generate structured output using JSON schema. Usage: generate_structured_output(prompt, schema)",
                is_dynamic=False,
                func=self.gemini_client.generate_structured_output,
            ),
            Tool(
                name="create_session",
                code="",
                doc="Create a new conversation session. Usage: create_session(session_id=None)",
                is_dynamic=False,
                func=self.gemini_client.create_session,
            ),
            # Enhanced file and directory tools
            Tool(
                name="search_in_file",
                code="",
                doc="Search for text in a file. Usage: search_in_file(file_path, search_term, case_sensitive=False)",
                is_dynamic=False,
                func=self.base_tools.search_in_file,
            ),
            Tool(
                name="replace_in_file",
                code="",
                doc="Replace text in a file. Usage: replace_in_file(file_path, old_text, new_text, case_sensitive=True)",
                is_dynamic=False,
                func=self.base_tools.replace_in_file,
            ),
            Tool(
                name="search_directory",
                code="",
                doc="Recursively search for text in all files within a directory. Usage: search_directory(directory, search_term, case_sensitive=False)",
                is_dynamic=False,
                func=self.base_tools.search_directory,
            ),
            Tool(
                name="find_files",
                code="",
                doc="Find files matching a glob pattern. Usage: find_files(pattern, directory='.')",
                is_dynamic=False,
                func=self.base_tools.find_files,
            ),
            Tool(
                name="create_archive",
                code="",
                doc="Create a zip or tar archive from a list of files. Usage: create_archive(archive_path, files, archive_type='zip')",
                is_dynamic=False,
                func=self.base_tools.create_archive,
            ),
            Tool(
                name="extract_archive",
                code="",
                doc="Extract a zip or tar archive. Usage: extract_archive(archive_path, extract_to=None)",
                is_dynamic=False,
                func=self.base_tools.extract_archive,
            ),
            Tool(
                name="read_json_file",
                code="",
                doc="Read and parse a JSON file. Usage: read_json_file(file_path)",
                is_dynamic=False,
                func=self.base_tools.read_json_file,
            ),
            Tool(
                name="write_json_file",
                code="",
                doc="Write data to a JSON file. Usage: write_json_file(file_path, data, indent=2)",
                is_dynamic=False,
                func=self.base_tools.write_json_file,
            ),
            Tool(
                name="read_csv_file",
                code="",
                doc="Read and parse a CSV file. Usage: read_csv_file(file_path, delimiter=',')",
                is_dynamic=False,
                func=self.base_tools.read_csv_file,
            ),
            Tool(
                name="write_csv_file",
                code="",
                doc="Write data to a CSV file. Usage: write_csv_file(file_path, data, headers=None, delimiter=',')",
                is_dynamic=False,
                func=self.base_tools.write_csv_file,
            ),
            Tool(
                name="run_linter",
                code="",
                doc="Run a Python linter (pylint) on a file or directory. Usage: run_linter(path)",
                is_dynamic=False,
                func=self.base_tools.run_linter,
            ),
            Tool(
                name="replace_in_multiple_files",
                code="",
                doc="Perform a search and replace operation across multiple files. Usage: replace_in_multiple_files(files, old_text, new_text, case_sensitive=True)",
                is_dynamic=False,
                func=self.base_tools.replace_in_multiple_files,
            ),
            # Enhanced web search tools
            Tool(
                name="enhanced_web_search",
                code="",
                doc="Perform web search with adaptive result length based on query complexity. Usage: enhanced_web_search(query, adaptive=True)",
                is_dynamic=False,
                func=self.gemini_client.enhanced_web_search,
            ),
            Tool(
                name="analyze_urls",
                code="",
                doc="Analyze multiple URLs and extract information. Usage: analyze_urls(urls)",
                is_dynamic=False,
                func=self.gemini_client.analyze_urls,
            ),
            # Additional file and system management tools
            Tool(
                name="get_file_info",
                code="",
                doc="Get detailed information about a file. Usage: get_file_info(file_path)",
                is_dynamic=False,
                func=self.base_tools.get_file_info,
            ),
            Tool(
                name="copy_file",
                code="",
                doc="Copy a file from source to destination. Usage: copy_file(source, destination)",
                is_dynamic=False,
                func=self.base_tools.copy_file,
            ),
            Tool(
                name="move_file",
                code="",
                doc="Move a file from source to destination. Usage: move_file(source, destination)",
                is_dynamic=False,
                func=self.base_tools.move_file,
            ),
            Tool(
                name="delete_file",
                code="",
                doc="Delete a file or directory. Usage: delete_file(file_path)",
                is_dynamic=False,
                func=self.base_tools.delete_file,
            ),
            Tool(
                name="create_directory",
                code="",
                doc="Create a directory. Usage: create_directory(directory_path)",
                is_dynamic=False,
                func=self.base_tools.create_directory,
            ),
            Tool(
                name="get_directory_size",
                code="",
                doc="Get the total size of a directory. Usage: get_directory_size(directory_path)",
                is_dynamic=False,
                func=self.base_tools.get_directory_size,
            ),
            Tool(
                name="find_large_files",
                code="",
                doc="Find files larger than specified size in MB. Usage: find_large_files(directory, min_size_mb=10.0)",
                is_dynamic=False,
                func=self.base_tools.find_large_files,
            ),
            Tool(
                name="get_system_disk_usage",
                code="",
                doc="Get disk usage information for all mounted drives. Usage: get_system_disk_usage()",
                is_dynamic=False,
                func=self.base_tools.get_system_disk_usage,
            ),
            Tool(
                name="get_process_info",
                code="",
                doc="Get information about running processes. Usage: get_process_info()",
                is_dynamic=False,
                func=self.base_tools.get_process_info,
            ),
            Tool(
                name="navigate_to_user_directories",
                code="",
                doc="Navigate to common user directories (Pictures, Desktop, etc.) based on OS. Usage: navigate_to_user_directories()",
                is_dynamic=False,
                func=self.base_tools.navigate_to_user_directories,
            ),
        ]

        # Add voice tools if enabled
        if self.voice_mode:
            voice_tools = [
                Tool(
                    name="speak",
                    code="",
                    doc="Convert text to speech. Usage: speak(text)",
                    is_dynamic=False,
                    func=self.voice_tools.speak,
                ),
                Tool(
                    name="listen",
                    code="",
                    doc="Listen for speech input. Usage: listen(timeout=5)",
                    is_dynamic=False,
                    func=self.voice_tools.listen,
                ),
                Tool(
                    name="test_voice_system",
                    code="",
                    doc="Test voice system functionality. Usage: test_voice_system()",
                    is_dynamic=False,
                    func=self.voice_tools.test_voice_system,
                ),
            ]
            core_tools.extend(voice_tools)

        # Register all tools
        for tool in core_tools:
            self.tool_manager.register_tool(tool)

        self.logger.success("Core tools registered successfully")

    def _load_tools_from_files(self):
        """Load tools from files in the tools directory."""
        self.logger.info("Loading tools from files...")

        for tool_file in settings.TOOLS_DIR.glob("*.py"):
            try:
                code = tool_file.read_text(encoding="utf-8")
                exec_context = {"system": self}
                exec(code, globals(), exec_context)

                tool_name = tool_file.stem
                func = exec_context.get(tool_name)

                if func:
                    tool = Tool(
                        name=tool_name,
                        code=code,
                        doc=f"Auto-loaded from {tool_file.name}",
                        is_dynamic=True,
                        last_used=0,
                        func=func,
                    )
                    self.tool_manager.register_tool(tool)
                    self.logger.success(
                        f"Tool '{tool_name}' loaded from {tool_file.name}"
                    )
            except Exception as e:
                self.logger.error(f"Failed to load tool from {tool_file}: {e}")

    def _get_initial_system_details(self):
        """Gather initial system information."""
        self.logger.info("Gathering system details...")

        system_info_result = self.base_tools.get_system_info()
        if system_info_result["success"]:
            self.context["system_info"] = system_info_result["output"]
        else:
            self.context["system_info"] = {"error": "Failed to gather system info"}

    def _gemini_request(self, prompt: str) -> Optional[str]:
        """Make a request to the Gemini API using the enhanced client."""
        try:
            # Add context to prompt
            context_summary = self.context_manager.get_context_summary()
            enhanced_prompt = f"""
{context_summary}

Current Request:
{prompt}
"""

            # Use Gemini client with context caching
            response = self.gemini_client.generate_with_context_caching(
                enhanced_prompt, "main_conversation"
            )

            if response.get("success"):
                # Add context entry
                self.context_manager.add_context_entry(
                    "ai_response",
                    response["text"],
                    {"usage": response.get("usage", {})},
                )
                return response["text"]
            else:
                self.logger.error(f"Gemini request failed: {response.get('error')}")
                return None

        except Exception as e:
            self.logger.error(f"Gemini request error: {e}", exc_info=True)
            return None

    def _construct_prompt(self) -> str:
        """Construct the prompt for the AI model."""
        tool_descriptions = "\n".join(
            [f"- {tool.name}: {tool.doc}" for tool in self.tool_manager.tools.values()]
        )

        # Build context string from context manager (actual context)
        context_string = ""
        if self.context_manager.current_project:
            # Get recent context entries from actual_context
            recent_context = self.context_manager.get_recent_context(
                20
            )  # Last 20 entries
            if recent_context:
                context_string += "**Recent Context (Last 20 entries):**\n"
                for entry in recent_context:
                    entry_type = entry.get("entry_type", "unknown")
                    content = entry.get("content", "")
                    timestamp = entry.get("timestamp", 0)
                    if timestamp:
                        dt = datetime.fromtimestamp(timestamp)
                        time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                        time_short = dt.strftime("%H:%M:%S")
                    else:
                        time_str = "Unknown"
                        time_short = "Unknown"

                    if entry_type == "user_request":
                        context_string += f"[{time_str}] User: {content}\n"
                    elif entry_type == "ai_response":
                        context_string += f"[{time_str}] AI: {content}\n"
                    elif entry_type == "tool_execution":
                        context_string += f"[{time_str}] Tool: {content}\n"
                    elif entry_type == "error":
                        context_string += f"[{time_str}] Error: {content}\n"
                    else:
                        context_string += f"[{time_str}] {entry_type}: {content}\n"
                context_string += "\n"

            # Add context summary if available
            if self.context_manager.current_project.context_summary:
                context_string += f"**Context Summary:**\n{self.context_manager.current_project.context_summary}\n\n"

        # Add relevant memories
        if self.context.get("initial_goal"):
            relevant_memories = self.memory_manager.recall(
                self.context["initial_goal"], top_k=5
            )
            for item in relevant_memories:
                context_string += f"- Memory: {item.content}\n"

        # Add execution history with detailed context
        execution_history_string = "\n".join(
            [
                f"- Step: {item.get('step', 'Unknown')}\n  Action: {item.get('action', 'Unknown')}\n  Success: {item.get('success', False)}\n  Output: {(item.get('output', '') or '')[:200]}...\n  Error: {item.get('error', 'None') or 'None'}\n"
                for item in self.context["execution_history"]
            ]
        )

        # Add last result context
        last_result_context = ""
        if "last_result" in self.context:
            last_result = self.context["last_result"]
            full_result = last_result.get("full_result", {})

            # Show additional result fields if available
            additional_info = ""
            if full_result:
                for key, value in full_result.items():
                    if key not in ["success", "error", "output"] and value:
                        additional_info += f"- {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}\n"

            last_result_context = f"""
**Last Tool Execution:**
- Action: {last_result.get('action', 'Unknown')}
- Success: {last_result.get('success', False)}
- Output: {last_result.get('output', '') or 'None'}
- Error: {last_result.get('error', 'None') or 'None'}
{additional_info}"""

        # Add last error context
        last_error_context = ""
        if "last_error" in self.context:
            last_error = self.context["last_error"]
            last_error_context = f"""
**Last Error:**
- Action: {last_error.get('action', 'Unknown')}
- Error: {last_error.get('error', 'Unknown') or 'None'}
- Args: {last_error.get('args', {})}
"""

        system_info = json.dumps(self.context["system_info"], indent=2)
        initial_goal_text = (
            f"The user's original goal was: {self.context['initial_goal']}\n"
            if self.context.get("initial_goal")
            else ""
        )

        prompt_template = f"""
You are a highly intelligent, autonomous, and self-improving AI system. Your primary objective is to complete the user's request. You have full access to the system and a comprehensive set of tools to achieve any task.

Your core process is as follows:
1. **Analyze**: Carefully break down the user's request.
2. **Recall & Contextualize**: Use your long-term memory and recent conversation history to retrieve relevant context.
3. **Plan**: Create a detailed, step-by-step plan. Each step must be a single tool call.
4. **Execute**: Execute the plan, one step at a time.
5. **Reflect & Improve**: Analyze the output of each tool. If an error occurs, self-heal by creating a new plan to diagnose and fix the issue.

**Your Guiding Principles:**
- Be completely dynamic and adaptive - never assume specific paths, directories, or file locations exist
- Explore and discover everything dynamically by starting from the current directory
- Use `run_shell` to execute system commands (with proper quoting for file paths with spaces)
- Use `read_file` and `write_file` to interact with the file system
- Use `list_dir` to explore the file system and discover what actually exists
- Use `change_dir` to navigate to directories you discover
- Use `find_files` to search for files by pattern (e.g., "*.png", "*.jpg")
- Use `analyze_image` to analyze any image file you discover
- Use `read_screen` to capture and analyze the current screen
- Use `analyze_screen_actions` to get actionable steps with coordinates for completing tasks
- Use `click_screen` to click at specific screen coordinates (x, y)
- Use `scroll_screen` to scroll at coordinates (x, y, direction, amount)
- Use `move_mouse` to move mouse to coordinates (x, y)
- Use `drag_mouse` to drag from (x1, y1) to (x2, y2)
- Use `type_text` to type text at coordinates (text, x, y)
- Use `press_key` to press key combinations
- Use `get_mouse_position` to get current mouse coordinates
- Use `run_shell` to execute system commands and open applications
- Use `install_package` to install Python packages in virtual environment
- Use `install_system_package` to install system packages using package manager
- Use `check_system_dependency` to check if system tools are installed and get installation instructions
- Use `search_directory` to search for text content across multiple files
- Use `enhanced_web_search` for comprehensive web searches
- When the entire task is complete, you MUST use the `complete_task` tool with a summary message
- **CRITICAL**: Do not repeat a failed step. If a step fails, your next plan must be to diagnose the failure and propose a new approach
- **DYNAMIC**: Never hardcode paths - always explore and discover what exists
- **ADAPTIVE**: Work with whatever you find, don't assume anything exists
- **EXPLORATION**: Start from current directory and systematically explore to find what you need
- **NAVIGATION**: When you find a directory you need to explore, use `change_dir` to navigate to it, then use `list_dir` to see its contents

**File and Directory Creation Rules:**
{self._get_system_rules()}

**Cross-Platform Directory Navigation Strategy:**
ALWAYS start your exploration from the OS root directories and work your way down:

**Windows:**
- Start from: `C:\\Users\\` then navigate to `[username]\\Pictures\\Screenshots`
- Alternative paths: `C:\\Users\\[username]\\Desktop\\Screenshots`
- Use: `change_dir("C:\\Users")` → `list_dir(".")` → `change_dir("[username]")` → `change_dir("Pictures")` → `change_dir("Screenshots")`

**Linux:**
- Start from: `/home/` then navigate to `[username]/Pictures/Screenshots`
- Alternative paths: `/home/[username]/Desktop/screenshots`
- Use: `change_dir("/home")` → `list_dir(".")` → `change_dir("[username]")` → `change_dir("Pictures")` → `change_dir("Screenshots")`

**macOS:**
- Start from: `/Users/` then navigate to `[username]/Pictures/Screenshots`
- Alternative paths: `/Users/[username]/Desktop/Screenshots`
- Use: `change_dir("/Users")` → `list_dir(".")` → `change_dir("[username]")` → `change_dir("Pictures")` → `change_dir("Screenshots")`

**Navigation Rules:**
1. ALWAYS start from the OS root directory (C:\\Users, /home, /Users)
2. Use `navigate_to_user_directories` to automatically find and navigate to common user directories
3. Use `list_dir` to see what's available at each level
4. Navigate step by step: root → username → Pictures → Screenshots
5. If a directory doesn't exist, try alternative paths
6. Use `find_files` with patterns like "*.png", "*.jpg" to locate images

**Quick Start for Image/Screenshot Tasks:**
1. Use `navigate_to_user_directories` to automatically find Pictures/Screenshots directories
2. Use `list_dir` to see what's in the current directory
3. Use `find_files` with "*.png" or "*.jpg" to find image files
4. Use `analyze_image` to analyze any found images or `read_screen` to capture and analyze current screen

**Key Navigation Strategy:**
1. First, use `list_dir` to see what's in the current directory
2. If you see a directory you need to explore (like "screenshots", "Pictures", "Desktop"), use `change_dir` to navigate to it
3. Then use `list_dir` again to see what's inside that directory
4. Use `find_files` with patterns like "*.png", "*.jpg", "*.jpeg", "*.gif" to find image files
5. Use `analyze_image` to analyze any image you find or `read_screen` to capture and analyze current screen
6. If you need to find specific content, use `search_directory` to search across multiple files

**Dynamic Discovery Examples:**
- For screenshots: Look in Desktop, Pictures, Downloads, or any folder with "screenshot" in the name
- For images: Use `find_files` with patterns like "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp"
- For documents: Look in Documents, Desktop, or use `find_files` with "*.pdf", "*.doc", "*.txt"
- For code: Look for folders with "src", "code", "project" in the name, or use `find_files` with "*.py", "*.js", "*.html"

**Available Tools:**
{tool_descriptions}

**System Information:**
{system_info}

**Conversation & Memory Context:**
{context_string}

**Execution History (to prevent repeated failures):**
{execution_history_string}

{last_result_context}

{last_error_context}

{initial_goal_text}
Current Status:
{self.context.get('status', 'No specific status. Ready for a new task or step.')}

**IMPORTANT CONTEXT AWARENESS:**
- Always check the "Last Tool Execution" section to see what the previous step did
- If the last step was successful, use its output to inform your next action
- If the last step failed, analyze the error and create a plan to fix it
- Never repeat the same failed step - always try a different approach
- Use the execution history to understand what has been tried before

**INTELLIGENT DYNAMIC WORKFLOW:**
- ALWAYS use `check_browser_status` first to see what applications are running
- If an application is running but not active, use `focus_window` to activate it
- Only open new application instances if no relevant app is running
- Use `run_shell` to execute system commands and open applications
- For web tasks: check browser status first, then either activate existing or open new
- Use `read_screen` to see what's actually on screen
- NEVER get stuck in loops - if a tool fails 2-3 times, try a completely different approach
- Always adapt to what's actually present, not what you assume should be there

**INTELLIGENT WINDOW MANAGEMENT:**
1. Use `get_active_window` to see what window is currently focused
2. Use `get_all_windows` to see all available windows and their details (shows ID, name, class)
3. Use `focus_window` to focus a specific window by ID, name, or class
4. Use `bring_window_to_front` to intelligently bring relevant windows to front
5. Always check window status before taking screenshots to ensure correct content
6. **IMPORTANT**: When you get window information, USE IT! Don't call get_all_windows repeatedly
7. Look for browser windows by checking window names/classes for terms like 'browser', 'web', 'mozilla', 'webkit'
8. Use the window ID or exact name from get_all_windows output to focus the correct window

**DYNAMIC APPLICATION HANDLING:**
1. Use `check_browser_status` to see what applications are running
2. If applications are running but not active, use window management tools to focus them
3. If no relevant app is running, use `run_shell_async` for blocking commands
4. Use `run_shell_async` for any command that might block (GUI apps, etc.)
5. Use `run_shell` for quick commands that return immediately
6. Use `interact_with_process` to check status, send input, or get output from running processes
7. Wait for applications to load before taking screenshots
8. Use `read_screen` to understand the current state
9. Adapt to any application or website dynamically
10. If something fails, try a completely different approach

**INTELLIGENT PROJECT HANDLING:**
1. Navigate to project directories using `change_dir` when working on specific projects
2. Use appropriate tools and environments for the task at hand
3. Handle directory paths with spaces by using proper quoting in commands
4. Work from the appropriate directory for the specific task
5. Adapt to whatever project structure you find

**INTELLIGENT TASK EXECUTION:**
- Understand the user's intent from their request
- Break down complex tasks into logical, sequential steps
- Use screen analysis to determine what actions are needed
- Adapt the approach based on what's actually visible
- Complete the full intent, not just the first obvious step
- Verify results by checking the actual outcome
- If a step fails, analyze the error and try a different approach
- Don't repeat the same failed action - always try something new
- Use context from previous successful steps to inform next actions

**INTELLIGENT ERROR HANDLING:**
- When errors occur, analyze the specific error message
- Try alternative approaches rather than repeating failed actions
- Use context from previous successful steps to guide error resolution
- If a directory/file already exists, handle it intelligently (remove and recreate, or use different name)
- If a command is not found, try alternative commands or install missing dependencies
- Always provide meaningful progress updates to the user
- If coordinate errors occur, try typing without coordinates or using fallback coordinates
- If window focus fails, try different window identifiers or open new applications
- Complete the full task intent, not just the first step

**ASYNC PROCESS MANAGEMENT:**
- `run_shell_async(command, timeout=0)` - Start process in background, returns process_id immediately
- `interact_with_process(process_id, "status")` - Check if process is still running
- `interact_with_process(process_id, "get_output")` - Get real-time output from process
- `interact_with_process(process_id, "send_input", data)` - Send input to process
- `interact_with_process(process_id, "kill")` - Terminate the process

**DYNAMIC INTERACTION RULES:**
- Analyze the current screen state before taking any action
- Identify clickable elements based on what's actually visible
- Use intelligent window management - bring windows to front when needed
- Wait for actions to complete before taking new screenshots
- Take screenshots only when necessary to verify state changes
- Adapt to any application or website, not just specific ones
- Understand the complete intent and execute accordingly

**INTELLIGENT TASK COMPLETION:**
- Analyze what the user actually wants to accomplish
- Determine if the task requires GUI interaction or just opening an application
- Use context and screen analysis to understand what needs to be done
- Complete the full intent of the task, not just the first step
- Verify completion by checking the actual result, not assumptions

**INTELLIGENT SCREENSHOT USAGE:**
- Take ONE screenshot to understand the current state
- Only take additional screenshots if the previous action failed or state is unclear
- Wait 2-3 seconds after actions before taking verification screenshots
- Avoid redundant screenshot analysis - trust successful actions

Your plan MUST be a JSON object with a single step, using the following structure:
{{
    "plan": [
        {{
            "step": "string describing the single step",
            "action": "tool_name",
            "args": {{ "arg1": "value1", "arg2": "value2" }}
        }}
    ],
    "comment": "brief summary of the plan"
}}

Always return a valid JSON object.
"""
        return prompt_template

    def process_request(self, user_input: str):
        """Process a user request with monitoring and error handling."""
        start_time = time.time()
        success = False
        error_type = None

        try:
            self.context["initial_goal"] = user_input
            self.context["status"] = f"Initial user request: {user_input}"
            self.context["conversation_history"].append({"user": user_input, "ai": ""})

            # Add to context manager
            self.context_manager.add_context_entry("user_request", user_input)

            # Reset execution history for new task
            self.context["execution_history"].clear()

            # Record request start
            metrics_collector.increment_counter("requests_started")

            self.logger.debug(f"Processing Request: {user_input}")
            print(f"\n🔄 Processing: {user_input}")

            while True:
                prompt = self._construct_prompt()
                response_text = self._gemini_request(prompt)

                if not response_text:
                    self.logger.error("Failed to get response from AI model")
                    break

                # Update conversation history
                if self.context["conversation_history"]:
                    last_conv = self.context["conversation_history"][-1]
                    last_conv["ai"] = response_text.strip()
                    self.context["conversation_history"][-1] = last_conv

                # Parse response
                plan_data = self._parse_response(response_text)
                if not plan_data:
                    self.logger.error("Failed to parse AI response")
                    break

                # Execute plan
                if self._execute_plan(plan_data):
                    success = True
                    break

        except Exception as e:
            error_type = type(e).__name__
            self.logger.error(f"Error processing request: {e}")
            success = False

        finally:
            # Record metrics
            duration = time.time() - start_time
            record_request_metric(duration, success, error_type)

            if success:
                metrics_collector.increment_counter("requests_completed")
            else:
                metrics_collector.increment_counter("requests_failed")

    def _parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse AI response and extract plan with comprehensive validation."""
        plan_data = None

        # Try to extract JSON from markdown code block
        json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                plan_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON from markdown: {e}")

        # Try to parse entire response as JSON if markdown parsing failed
        if not plan_data:
            try:
                plan_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON: {e}")
                return None

        # Comprehensive plan validation
        if plan_data:
            validation_result = self._validate_plan(plan_data)
            if not validation_result["valid"]:
                self.logger.error(
                    f"Plan validation failed: {validation_result['error']}"
                )
                return None

        return plan_data

    def _validate_plan(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive plan validation."""
        try:
            # Check if plan exists and is a list
            if "plan" not in plan_data:
                return {"valid": False, "error": "Missing 'plan' field"}

            if not isinstance(plan_data["plan"], list):
                return {"valid": False, "error": "'plan' must be an array"}

            if len(plan_data["plan"]) == 0:
                return {"valid": False, "error": "Plan cannot be empty"}

            # Validate each step in the plan
            for i, step in enumerate(plan_data["plan"]):
                if not isinstance(step, dict):
                    return {"valid": False, "error": f"Step {i+1} must be an object"}

                # Check required fields
                required_fields = ["step", "action", "args"]
                for field in required_fields:
                    if field not in step:
                        return {
                            "valid": False,
                            "error": f"Step {i+1} missing required field: {field}",
                        }

                # Validate action field
                if not isinstance(step["action"], str) or not step["action"].strip():
                    return {
                        "valid": False,
                        "error": f"Step {i+1} action must be a non-empty string",
                    }

                # Validate args field
                if not isinstance(step["args"], dict):
                    return {
                        "valid": False,
                        "error": f"Step {i+1} args must be an object",
                    }

                # Check if tool exists
                tool_name = step["action"]
                if not self.tool_manager.tool_exists(tool_name):
                    return {
                        "valid": False,
                        "error": f"Step {i+1} references unknown tool: {tool_name}",
                    }

                # Check for potentially problematic patterns (but allow legitimate retries)
                step_desc = step.get("step", "").lower()
                # Only block if it's clearly a repetitive retry without new information
                if (
                    ("again" in step_desc or "retry" in step_desc)
                    and "previous" in step_desc
                    and not any(word in step_desc for word in ["new", "different", "alternative", "fix", "resolve", "install", "remove", "check", "click", "type", "search", "play", "video"])
                ):
                    return {
                        "valid": False,
                        "error": f"Step {i+1} contains repetitive retry language - try a different approach",
                    }

                # Check for window management loops
                if tool_name in ["focus_window", "bring_window_to_front", "get_all_windows"] and i > 0:
                    prev_action = plan_data["plan"][i - 1].get("action", "")
                    if prev_action in [
                        "focus_window",
                        "bring_window_to_front",
                        "get_active_window",
                        "get_all_windows",
                    ]:
                        return {
                            "valid": False,
                            "error": f"Step {i+1} creates window management loop - avoid consecutive window operations",
                        }
                
                # Check for excessive get_all_windows calls
                if tool_name == "get_all_windows":
                    window_calls = sum(1 for step in plan_data["plan"][:i+1] if step.get("action") == "get_all_windows")
                    if window_calls > 2:
                        return {
                            "valid": False,
                            "error": f"Step {i+1} - too many get_all_windows calls ({window_calls}), use the window information already obtained",
                        }

            # Validate comment field if present
            if "comment" in plan_data and not isinstance(plan_data["comment"], str):
                return {"valid": False, "error": "Comment must be a string"}

            return {"valid": True, "error": None}

        except Exception as e:
            return {"valid": False, "error": f"Validation error: {e}"}

    def _execute_plan(self, plan_data: Dict[str, Any]) -> bool:
        """Execute a plan step with enhanced error handling and reliability."""
        try:
            plan = plan_data.get("plan")
            if not isinstance(plan, list) or not plan:
                self.logger.error("Invalid plan structure")
                return True

            step_data = plan[0]
            action = step_data.get("action")
            args = step_data.get("args", {})
            comment = plan_data.get("comment", "No comment provided")

            # Validate step data before execution
            if not action or not isinstance(action, str):
                self.logger.error("Invalid action in plan step")
                return True

            if not isinstance(args, dict):
                self.logger.error("Invalid args in plan step")
                return True

            self.logger.step(1, 1, comment)

            # Store plan in context
            self.context["last_plan"] = {
                "action": action,
                "args": args,
                "comment": comment,
                "timestamp": time.time(),
            }

            # Check for repeated failed steps
            step_identifier = json.dumps(step_data, sort_keys=True)
            if self.execution_history.has_failed_before(step_identifier):
                self.logger.error(
                    "This step has previously failed. Skipping to avoid loops."
                )
                self.context["status"] = (
                    f"Previous plan '{comment}' has failed before. Please generate a different plan."
                )
                return False

            # Execute tool
            if action not in self.tool_manager.tools:
                error_msg = f"Tool '{action}' not found"
                self.logger.error(error_msg)
                self.context["status"] = f"Plan execution failed: {error_msg}"
                self.execution_history.log_execution(
                    step_identifier, comment, "failed", error_msg
                )
                return False

            tool = self.tool_manager.tools[action]

            # Execute tool with retry logic and comprehensive error handling
            max_retries = 3
            retry_delay = 1.0
            tool_start_time = time.time()

            for attempt in range(max_retries):
                try:
                    self.tool_manager.update_tool_usage(action)
                    self.logger.debug(
                        f"🔧 Executing tool: {action} (attempt {attempt + 1}/{max_retries})"
                    )
                    self.logger.debug(f"📋 Arguments: {json.dumps(args, indent=2)}")

                    # Validate tool function exists
                    if not hasattr(tool, "func") or not callable(tool.func):
                        error_msg = f"Tool '{action}' has no callable function"
                        self.logger.error(error_msg)
                        self.context["status"] = f"Tool execution failed: {error_msg}"
                        self.execution_history.log_execution(
                            step_identifier, comment, "failed", error_msg
                        )
                        return False

                    result = tool.func(**args)

                    # Record tool execution metrics
                    tool_duration = time.time() - tool_start_time
                    tool_success = (
                        result.get("success", False)
                        if isinstance(result, dict)
                        else False
                    )
                    record_tool_metric(action, tool_duration, tool_success)

                    # Validate result structure
                    if not isinstance(result, dict):
                        error_msg = f"Tool '{action}' returned invalid result type: {type(result)}"
                        self.logger.error(error_msg)
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        else:
                            self.context["status"] = (
                                f"Tool execution failed: {error_msg}"
                            )
                            self.execution_history.log_execution(
                                step_identifier, comment, "failed", error_msg
                            )
                            return False

                    break  # Success, exit retry loop

                except TypeError as e:
                    error_msg = f"Tool '{action}' argument error: {e}"
                    self.logger.error(error_msg)
                    if "unexpected keyword argument" in str(e):
                        # Don't retry argument errors
                        self.context["status"] = f"Tool execution failed: {error_msg}"
                        self.execution_history.log_execution(
                            step_identifier, comment, "failed", error_msg
                        )
                        return False
                    elif attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        self.context["status"] = f"Tool execution failed: {error_msg}"
                        self.execution_history.log_execution(
                            step_identifier, comment, "failed", error_msg
                        )
                        return False

                except Exception as e:
                    error_msg = f"Tool '{action}' execution error: {e}"
                    self.logger.error(error_msg)
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        self.context["status"] = f"Tool execution failed: {error_msg}"
                        self.execution_history.log_execution(
                            step_identifier, comment, "failed", error_msg
                        )
                        return False

            if result.get("success"):
                self.logger.debug(f"✅ Tool '{action}' completed successfully")
            else:
                self.logger.debug(
                    f"❌ Tool '{action}' failed: {result.get('error', 'Unknown error')}"
                )

            # Dynamically extract all meaningful content from result
            # Look for common output field names in order of preference
            output_fields = [
                "output",
                "text",
                "result",
                "content",
                "data",
                "message",
                "response",
            ]
            output_text = ""
            for field in output_fields:
                if field in result and result[field]:
                    output_text = str(result[field])
                    break

            # If no standard output field found, look for any string value
            if not output_text:
                for key, value in result.items():
                    if (
                        isinstance(value, str)
                        and value.strip()
                        and key not in ["success", "error"]
                    ):
                        output_text = value
                        break

            # If still no output, try to convert the entire result to string
            if not output_text and result:
                # Exclude non-essential fields from the string representation
                filtered_result = {
                    k: v
                    for k, v in result.items()
                    if k
                    not in ["success", "error", "timestamp", "usage", "usage_metadata"]
                }
                if filtered_result:
                    output_text = str(filtered_result)

            # Store detailed result in context with full result data
            self.context["last_result"] = {
                "action": action,
                "args": args,
                "success": result.get("success", False),
                "output": output_text,
                "error": result.get("error", ""),
                "full_result": result,  # Store the complete result for reference
                "timestamp": time.time(),
            }

            # Add tool execution to context manager
            tool_execution_content = (
                f"Executed '{action}' with args: {json.dumps(args, indent=2)}"
            )
            if result.get("success"):
                tool_execution_content += f"\nResult: {output_text[:500]}{'...' if len(output_text) > 500 else ''}"
            else:
                tool_execution_content += (
                    f"\nError: {result.get('error', 'Unknown error')}"
                )

            self.context_manager.add_context_entry(
                "tool_execution",
                tool_execution_content,
                {
                    "action": action,
                    "args": args,
                    "success": result.get("success", False),
                },
            )

            # Add to execution history in context
            execution_entry = {
                "step": comment,
                "action": action,
                "args": args,
                "success": result.get("success", False),
                "output": output_text,
                "error": result.get("error", ""),
                "full_result": result,  # Store the complete result for reference
                "timestamp": time.time(),
            }
            self.context["execution_history"].append(execution_entry)

            if output_text == "TASK_COMPLETED_SIGNAL":
                print(f"\n{'🎉'*20}")
                print(f"🎉 TASK COMPLETED SUCCESSFULLY! 🎉")
                print(f"{'🎉'*20}\n")
                self.logger.success("Task completed successfully!")
                self.context["status"] = "Task completed successfully."
                return True

            if result.get("success"):
                # Display structured output without duplicating the full result
                print(f"\n{'='*60}")
                print(f"✅ STEP COMPLETED: {comment}")
                print(f"{'='*60}")
                print(f"📋 Tool: {action}")

                # Use summary if available, otherwise show truncated output
                summary = result.get("summary", "")
                if summary:
                    print(f"📊 Result: {summary}")
                elif action == "read_screen":
                    print(
                        f"📊 Result: Screen analyzed successfully - see logs for full details"
                    )
                elif action in ["get_all_windows", "get_active_window", "focus_window"]:
                    # For window tools, show full output as it's essential for debugging
                    print(f"📊 Result: {output_text}")
                else:
                    # For other tools, show truncated output
                    truncated_output = (
                        output_text[:200] + "..."
                        if len(output_text) > 200
                        else output_text
                    )
                    print(f"📊 Result: {truncated_output}")

                print(f"{'='*60}\n")

                # Log the full output for debugging and context
                self.logger.success(f"Step completed. Full output:\n{output_text}")
                self.memory_manager.remember(
                    f"Executed '{action}' successfully. Output: {output_text}",
                    {"type": "tool_output", "tool": action},
                )
                self.context["status"] = (
                    f"Previous step '{comment}' completed successfully. Output: {output_text[:100]}..."
                )
                self.execution_history.log_execution(
                    step_identifier, comment, "success"
                )
            else:
                error_msg = result.get("error")
                self.logger.error(f"Step failed. Error:\n{error_msg}")

                # Store error in context for better error resolution
                self.context["last_error"] = {
                    "action": action,
                    "args": args,
                    "error": error_msg,
                    "timestamp": time.time(),
                }

                # Add error to context manager
                error_content = f"Tool '{action}' failed with args: {json.dumps(args, indent=2)}\nError: {error_msg}"
                self.context_manager.add_context_entry(
                    "error",
                    error_content,
                    {"action": action, "args": args, "error": error_msg},
                )

                # Attempt smart error resolution
                if self._attempt_error_resolution(
                    action, args, error_msg, step_identifier, comment
                ):
                    return True

                self.memory_manager.remember(
                    f"Execution of '{action}' failed: {error_msg}",
                    {"type": "tool_error", "tool": action},
                )
                self.context["status"] = (
                    f"Previous step failed: {error_msg}. Please generate a plan to resolve this."
                )
                self.execution_history.log_execution(
                    step_identifier, comment, "failed", error_msg
                )

            return False

        except Exception as e:
            error_msg = f"Unexpected error during execution: {e}"
            self.logger.error(error_msg, exc_info=True)

            # Store exception in context
            self.context["last_error"] = {
                "action": action,
                "args": args,
                "error": error_msg,
                "timestamp": time.time(),
            }

            self.memory_manager.remember(
                f"Unexpected error with '{action}': {e}", {"type": "tool_error"}
            )
            self.context["status"] = (
                f"Unexpected error during execution: {e}. Please diagnose and fix this issue."
            )
            self.execution_history.log_execution(
                step_identifier, comment, "failed", error_msg
            )
            return False

    def run(self):
        """Main run loop for the AI system."""
        if self.voice_mode:
            self.voice_tools.speak(
                "Hello, I am a self-improving AI system. How can I help you today?"
            )
        else:
            self.logger.info("AI System ready. Type your request or 'exit' to quit.")

        while self.active:
            if self.voice_mode:
                listen_result = self.voice_tools.listen()
                if not listen_result["success"]:
                    self.logger.error(f"Voice input failed: {listen_result['error']}")
                    user_input = input("\n> ")
                else:
                    user_input = listen_result["output"]
            else:
                user_input = input("\n> ")

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "goodbye"]:
                if self.voice_mode:
                    self.voice_tools.speak("Goodbye!")
                else:
                    self.logger.info("Goodbye!")
                self.active = False
                continue

            self.process_request(user_input)

    def shutdown(self):
        """Shutdown the AI system and clean up resources."""
        self.logger.info("Shutting down AI system...")
        self.active = False

        # Stop health server
        if self.health_server:
            self.health_server.stop()
            self.logger.info("Health server stopped")

        self.memory_manager.close()
        self.tool_manager.close()
        self.execution_history.close()
        self.context_manager.cleanup_old_projects()
        self.logger.success("AI system shutdown complete")

    def _attempt_error_resolution(
        self,
        tool_name: str,
        args: Dict[str, Any],
        error_msg: str,
        step_identifier: str,
        comment: str,
    ) -> bool:
        """Attempt to automatically resolve common errors with enhanced context awareness."""
        self.logger.info(f"🔧 Attempting error resolution for: {error_msg}")

        # Check if we have context from previous successful steps
        last_successful_result = None
        for entry in reversed(self.context["execution_history"]):
            if entry.get("success", False):
                last_successful_result = entry
                break

        # Directory navigation errors - use context from previous successful list_dir
        if "Directory not found" in error_msg and tool_name == "change_dir":
            if (
                last_successful_result
                and last_successful_result.get("action") == "list_dir"
            ):
                # Extract directory contents from previous successful list_dir
                output = last_successful_result.get("output", "")
                if output:
                    files = output.split("\n")
                    # Look for directories that might match what we're trying to access
                    target_dir = args.get("directory", "")
                    for file in files:
                        if file.strip() and target_dir.lower() in file.lower():
                            # Found a matching directory, try to navigate to it
                            full_path = file.strip()
                            self.logger.info(
                                f"🔧 Context-based resolution: Found matching directory {full_path}"
                            )

                            # Try to navigate to the full path
                            change_result = self.base_tools.change_dir(full_path)
                            if change_result.get("success"):
                                self.logger.success(
                                    f"✅ Context-based error resolved! Navigated to: {full_path}"
                                )
                                self.memory_manager.remember(
                                    f"Context-based directory resolution: {target_dir} -> {full_path}",
                                    {"type": "context_resolution"},
                                )
                                self.context["status"] = (
                                    f"Navigated to {full_path} using context from previous step"
                                )
                                return True

        # Django command not found errors
        if "not found" in error_msg.lower() and "django" in error_msg.lower():
            self.logger.info("🔧 Django command not found, installing Django...")
            install_result = self.base_tools.install_package("django")
            if install_result.get("success"):
                self.logger.success("✅ Django installed successfully")
                # Try the original command again
                if tool_name == "run_shell":
                    command = args.get("command", "")
                    if "django-admin" in command:
                        # Replace django-admin with python3 -m django
                        new_command = command.replace(
                            "django-admin", "python3 -m django"
                        )
                        new_args = args.copy()
                        new_args["command"] = new_command
                        result = self.base_tools.run_shell(**new_args)
                        if result.get("success"):
                            self.logger.success(
                                "✅ Django command executed successfully after installation"
                            )
                            self.context["status"] = (
                                "Django project created successfully"
                            )
                            return True
            return False

        # Django project already exists errors
        if "already exists" in error_msg.lower() and "django" in error_msg.lower():
            self.logger.info("🔧 Django project already exists, handling conflict...")
            if tool_name == "run_shell":
                command = args.get("command", "")
                if "django-admin startproject" in command:
                    # Extract project name from command
                    parts = command.split()
                    project_name = parts[-1] if parts else "project"
                    
                    # Try to create with a different name
                    new_name = f"{project_name}_new"
                    new_command = command.replace(project_name, new_name)
                    new_args = args.copy()
                    new_args["command"] = new_command
                    
                    self.logger.info(f"🔧 Trying to create project with name: {new_name}")
                    result = self.base_tools.run_shell(**new_args)
                    if result.get("success"):
                        self.logger.success(f"✅ Django project created with name: {new_name}")
                        self.context["status"] = f"Django project '{new_name}' created successfully"
                        return True
                    
                    # If that fails, try to remove existing directory and recreate
                    self.logger.info(f"🔧 Removing existing directory and recreating...")
                    remove_cmd = f"rm -rf {project_name}"
                    remove_result = self.base_tools.run_shell({"command": remove_cmd})
                    if remove_result.get("success"):
                        # Now try original command
                        result = self.base_tools.run_shell(args)
                        if result.get("success"):
                            self.logger.success("✅ Django project created after removing existing directory")
                            self.context["status"] = "Django project created successfully"
                            return True
            return False

        # Virtual environment activation errors
        if "source" in error_msg.lower() and "not found" in error_msg.lower():
            self.logger.info(
                "🔧 Virtual environment activation failed, trying direct execution..."
            )
            if tool_name == "run_shell":
                command = args.get("command", "")
                if "source venv/bin/activate" in command:
                    # Extract the command after activation
                    parts = command.split("&&")
                    if len(parts) > 1:
                        new_command = parts[1].strip()
                        new_args = args.copy()
                        new_args["command"] = new_command
                        result = self.base_tools.run_shell(**new_args)
                        if result.get("success"):
                            self.logger.success(
                                "✅ Command executed successfully without virtual environment"
                            )
                            self.context["status"] = "Command executed successfully"
                            return True
            return False

        # XTest coordinate errors - try alternative approach
        if "X Error of failed request" in error_msg and "BadValue" in error_msg:
            self.logger.info("🔧 XTest coordinate error detected, trying alternative approach...")
            if tool_name in ["type_text", "click_screen"]:
                # Try without coordinates first
                if tool_name == "type_text":
                    new_args = {"text": args.get("text", "")}
                    result = self.base_tools.type_text(**new_args)
                    if result.get("success"):
                        self.logger.success("✅ Text typed successfully without coordinates")
                        self.context["status"] = "Text typed successfully using alternative method"
                        return True
                elif tool_name == "click_screen":
                    # Try clicking at center of screen as fallback
                    new_args = {"x": 960, "y": 540, "button": args.get("button", "left")}
                    result = self.base_tools.click_screen(**new_args)
                    if result.get("success"):
                        self.logger.success("✅ Clicked at screen center as fallback")
                        self.context["status"] = "Clicked successfully using fallback coordinates"
                        return True
            return False

        # Directory/file already exists errors (general)
        if "already exists" in error_msg.lower():
            self.logger.info("🔧 Handling 'already exists' error...")
            if tool_name == "run_shell":
                command = args.get("command", "")
                # Check if it's a project creation command
                if any(cmd in command.lower() for cmd in ["startproject", "startapp", "create"]):
                    # Try to extract target name and create with different name
                    parts = command.split()
                    if len(parts) > 1:
                        target = parts[-1]
                        new_target = f"{target}_new"
                        new_command = command.replace(target, new_target)
                        self.logger.info(f"🔧 Trying with new name: {new_target}")
                        new_args = args.copy()
                        new_args["command"] = new_command
                        result = self.base_tools.run_shell(**new_args)
                        if result.get("success"):
                            self.logger.success(f"✅ Successfully created {new_target}")
                            self.context["status"] = f"Successfully created {new_target}"
                            return True
                # Check if it's a file/directory creation that can be removed
                elif any(cmd in command for cmd in ["mkdir", "touch"]):
                    # Extract the target name from command
                    parts = command.split()
                    if len(parts) > 1:
                        target = parts[-1]
                        # Try to remove and recreate
                        remove_cmd = f"rm -rf {target}"
                        self.logger.info(f"🔧 Removing existing {target} and recreating...")
                        remove_result = self.base_tools.run_shell({"command": remove_cmd})
                        if remove_result.get("success"):
                            # Now try original command
                            result = self.base_tools.run_shell(args)
                            if result.get("success"):
                                self.logger.success(f"✅ Successfully recreated {target}")
                                self.context["status"] = f"Successfully recreated {target}"
                                return True
            return False

        # File not found errors
        if "No such file or directory" in error_msg or "File not found" in error_msg:
            if tool_name == "analyze_image" and "image_path" in args:
                # Try to find the correct file path
                image_path = args["image_path"]
                directory = os.path.dirname(image_path)

                # List directory to find similar files
                list_result = self.base_tools.list_dir(directory)
                if list_result.get("success"):
                    files = list_result["output"].split("\n")
                    # Find files with similar names
                    similar_files = [
                        f
                        for f in files
                        if os.path.splitext(f)[1].lower()
                        in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"]
                    ]

                    if similar_files:
                        # Use the first similar file found
                        correct_path = os.path.join(directory, similar_files[0])
                        self.logger.info(f"🔧 Found similar file: {correct_path}")

                        # Retry with correct path
                        new_args = args.copy()
                        new_args["image_path"] = correct_path
                        tool = self.tool_manager.get_tool(tool_name)
                        if tool:
                            result = tool.func(**new_args)
                            if result.get("success"):
                                self.logger.success(
                                    f"✅ Error resolved! Using file: {correct_path}"
                                )
                                self.memory_manager.remember(
                                    f"Auto-resolved file path error: {image_path} -> {correct_path}",
                                    {"type": "error_resolution"},
                                )
                                self.context["status"] = (
                                    f"Step completed after error resolution"
                                )
                                self.execution_history.log_execution(
                                    step_identifier,
                                    comment,
                                    "success",
                                    result.get("output", ""),
                                )
                                return True

        # Smart navigation using context from previous successful list_dir
        elif "Directory not found" in error_msg and tool_name == "change_dir":
            if (
                last_successful_result
                and last_successful_result.get("action") == "list_dir"
            ):
                output = last_successful_result.get("output", "")
                if output:
                    files = output.split("\n")
                    target_dir = args.get("directory", "").lower()

                    # Look for exact matches first
                    for file in files:
                        if file.strip() and file.strip().lower() == target_dir:
                            full_path = file.strip()
                            self.logger.info(f"🔧 Exact match found: {full_path}")
                            change_result = self.base_tools.change_dir(full_path)
                            if change_result.get("success"):
                                self.logger.success(
                                    f"✅ Navigated to exact match: {full_path}"
                                )
                                self.context["status"] = f"Navigated to {full_path}"
                                return True

                    # Look for partial matches
                    for file in files:
                        if file.strip() and target_dir in file.strip().lower():
                            full_path = file.strip()
                            self.logger.info(f"🔧 Partial match found: {full_path}")
                            change_result = self.base_tools.change_dir(full_path)
                            if change_result.get("success"):
                                self.logger.success(
                                    f"✅ Navigated to partial match: {full_path}"
                                )
                                self.context["status"] = f"Navigated to {full_path}"
                                return True

        # Cross-platform directory discovery for images/screenshots
        elif any(
            keyword in comment.lower()
            for keyword in ["image", "picture", "screenshot", "photo"]
        ):
            if tool_name == "list_dir" and "directory" in args:
                directory = args["directory"]

                # Cross-platform common directories to try
                common_dirs = [
                    os.path.expanduser("~"),  # Home directory
                    os.path.join(os.path.expanduser("~"), "Desktop"),
                    os.path.join(os.path.expanduser("~"), "Pictures"),
                    os.path.join(os.path.expanduser("~"), "Downloads"),
                    os.path.join(os.path.expanduser("~"), "Documents"),
                    ".",  # Current directory
                    "..",  # Parent directory
                ]

                # Add platform-specific directories
                if sys.platform == "win32":
                    common_dirs.extend(
                        [
                            os.path.join(
                                os.path.expanduser("~"), "Pictures", "Screenshots"
                            ),
                            os.path.join(
                                os.path.expanduser("~"), "Desktop", "Screenshots"
                            ),
                        ]
                    )
                elif sys.platform == "darwin":  # macOS
                    common_dirs.extend(
                        [
                            os.path.join(
                                os.path.expanduser("~"), "Desktop", "Screenshots"
                            ),
                            os.path.join(
                                os.path.expanduser("~"), "Pictures", "Screenshots"
                            ),
                        ]
                    )
                else:  # Linux
                    common_dirs.extend(
                        [
                            os.path.join(
                                os.path.expanduser("~"), "Desktop", "screenshots"
                            ),
                            os.path.join(
                                os.path.expanduser("~"), "Pictures", "screenshots"
                            ),
                        ]
                    )

                # Try each directory
                for test_dir in common_dirs:
                    if os.path.exists(test_dir):
                        self.logger.info(f"🔧 Exploring directory: {test_dir}")
                        list_result = self.base_tools.list_dir(test_dir)
                        if list_result.get("success"):
                            files = list_result["output"].split("\n")
                            image_files = [
                                f
                                for f in files
                                if os.path.splitext(f)[1].lower()
                                in [
                                    ".png",
                                    ".jpg",
                                    ".jpeg",
                                    ".gif",
                                    ".bmp",
                                    ".webp",
                                    ".tiff",
                                ]
                            ]

                            if image_files:
                                # Select first image and analyze it
                                selected_image = os.path.join(test_dir, image_files[0])
                                self.logger.info(
                                    f"🔧 Cross-platform discovery: Found image {selected_image}"
                                )

                                # Use analyze_image tool
                                analyze_tool = self.tool_manager.get_tool(
                                    "analyze_image"
                                )
                                if analyze_tool:
                                    result = analyze_tool.func(
                                        image_path=selected_image,
                                        prompt="Describe this image in detail",
                                    )
                                    if result.get("success"):
                                        self.logger.success(
                                            f"✅ Cross-platform image analysis completed!"
                                        )
                                        self.memory_manager.remember(
                                            f"Cross-platform image analysis: {selected_image}",
                                            {"type": "cross_platform_analysis"},
                                        )
                                        self.context["status"] = (
                                            f"Cross-platform image analysis completed"
                                        )
                                        self.execution_history.log_execution(
                                            step_identifier,
                                            comment,
                                            "success",
                                            result.get("output", ""),
                                        )
                                        return True

                                # Also try to navigate to the directory for further exploration
                                change_result = self.base_tools.change_dir(test_dir)
                                if change_result.get("success"):
                                    self.logger.info(
                                        f"🔧 Navigated to directory: {test_dir}"
                                    )
                                    self.context["status"] = (
                                        f"Navigated to {test_dir} for further exploration"
                                    )
                                    return True

        # Dynamic content discovery - no assumptions about what exists
        elif any(
            keyword in comment.lower()
            for keyword in ["file", "document", "code", "project"]
        ):
            if tool_name == "list_dir" and "directory" in args:
                directory = args["directory"]
                # Dynamically explore and find relevant content
                list_result = self.base_tools.list_dir(directory)
                if list_result.get("success"):
                    files = list_result["output"].split("\n")

                    # Look for any directories or files that might be relevant
                    potential_targets = []
                    for f in files:
                        if f.strip():  # Skip empty lines
                            potential_targets.append(f)

                    # Try to find and process any relevant content
                    for target in potential_targets:
                        full_path = os.path.join(directory, target)

                        # Check if it's a directory
                        dir_result = self.base_tools.list_dir(full_path)
                        if dir_result.get("success"):
                            # It's a directory, explore it
                            sub_files = dir_result["output"].split("\n")
                            for sub_file in sub_files:
                                if sub_file.strip():
                                    sub_path = os.path.join(full_path, sub_file)
                                    # Check if it's an image file
                                    if os.path.splitext(sub_file)[1].lower() in [
                                        ".png",
                                        ".jpg",
                                        ".jpeg",
                                        ".gif",
                                        ".bmp",
                                        ".webp",
                                        ".tiff",
                                    ]:
                                        self.logger.info(
                                            f"🔧 Dynamic discovery: Found image {sub_path}"
                                        )

                                        # Use analyze_image tool
                                        analyze_tool = self.tool_manager.get_tool(
                                            "analyze_image"
                                        )
                                        if analyze_tool:
                                            result = analyze_tool.func(
                                                image_path=sub_path,
                                                prompt="Describe this image in detail",
                                            )
                                            if result.get("success"):
                                                self.logger.success(
                                                    f"✅ Dynamic image analysis completed!"
                                                )
                                                self.memory_manager.remember(
                                                    f"Dynamic image analysis: {sub_path}",
                                                    {"type": "dynamic_analysis"},
                                                )
                                                self.context["status"] = (
                                                    f"Dynamic image analysis completed"
                                                )
                                                self.execution_history.log_execution(
                                                    step_identifier,
                                                    comment,
                                                    "success",
                                                    result.get("output", ""),
                                                )
                                                return True
                        else:
                            # It's a file, check if it's an image
                            if os.path.splitext(target)[1].lower() in [
                                ".png",
                                ".jpg",
                                ".jpeg",
                                ".gif",
                                ".bmp",
                                ".webp",
                                ".tiff",
                            ]:
                                self.logger.info(
                                    f"🔧 Dynamic discovery: Found image {full_path}"
                                )

                                # Use analyze_image tool
                                analyze_tool = self.tool_manager.get_tool(
                                    "analyze_image"
                                )
                                if analyze_tool:
                                    result = analyze_tool.func(
                                        image_path=full_path,
                                        prompt="Describe this image in detail",
                                    )
                                    if result.get("success"):
                                        self.logger.success(
                                            f"✅ Dynamic image analysis completed!"
                                        )
                                        self.memory_manager.remember(
                                            f"Dynamic image analysis: {full_path}",
                                            {"type": "dynamic_analysis"},
                                        )
                                        self.context["status"] = (
                                            f"Dynamic image analysis completed"
                                        )
                                        self.execution_history.log_execution(
                                            step_identifier,
                                            comment,
                                            "success",
                                            result.get("output", ""),
                                        )
                                        return True

        # Dynamic error resolution - try different approaches
        elif "Directory not found" in error_msg or "Access denied" in error_msg:
            if tool_name == "list_dir" and "directory" in args:
                import os

                directory = args["directory"]

                # Try different approaches dynamically
                approaches = [
                    ("parent directory", os.path.dirname(directory)),
                    ("current directory", "."),
                    ("home directory", os.path.expanduser("~")),
                    ("project root", settings.BASE_DIR),
                ]

                for approach_name, test_dir in approaches:
                    if test_dir and test_dir != directory and os.path.exists(test_dir):
                        self.logger.info(f"🔧 Trying {approach_name}: {test_dir}")
                        new_args = args.copy()
                        new_args["directory"] = test_dir
                        tool = self.tool_manager.get_tool(tool_name)
                        if tool:
                            result = tool.func(**new_args)
                            if result.get("success"):
                                self.logger.success(
                                    f"✅ Error resolved! Found {approach_name}: {test_dir}"
                                )
                                self.memory_manager.remember(
                                    f"Auto-resolved directory error: {directory} -> {test_dir}",
                                    {"type": "error_resolution"},
                                )
                                self.context["status"] = (
                                    f"Step completed after error resolution"
                                )
                                self.execution_history.log_execution(
                                    step_identifier,
                                    comment,
                                    "success",
                                    result.get("output", ""),
                                )
                                return True

        self.logger.warning(f"❌ Could not auto-resolve error: {error_msg}")
        return False
