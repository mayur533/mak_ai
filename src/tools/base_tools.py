"""
Base tools for the AI Assistant System.
Contains core functionality tools that are always available.
"""

import ast
import io
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.config.settings import settings
from src.logging.logger import logger


class BaseTools:
    """Base tools that provide core functionality for the AI system."""

    def __init__(self, system=None):
        """Initialize base tools with optional system reference."""
        self.system = system
        self.logger = logger

    def run_shell(self, command: str, background: bool = None, timeout: int = None) -> Dict[str, Any]:
        """
        Execute a shell command with robust error handling and validation.
        Auto-detects GUI applications and runs them in background.
        Intelligently sets timeout based on command type.

        Args:
            command: The shell command to execute
            background: Force background execution (None=auto-detect, True=background, False=foreground)
            timeout: Timeout in seconds (None=auto-detect based on command)

        Returns:
            Dict with success status and output/error
        """
        try:
            # Validate input
            if not command or not command.strip():
                return {"success": False, "error": "Empty or invalid command provided"}

            command = command.strip()
            
            # Auto-detect GUI applications and background processes
            gui_apps = [
                'gedit', 'kate', 'nano', 'vim', 'emacs',  # Text editors
                'firefox', 'chrome', 'chromium', 'brave',  # Browsers
                'code', 'subl', 'atom',  # IDEs
                'nautilus', 'dolphin', 'thunar',  # File managers
                'gimp', 'inkscape', 'blender',  # Graphics
                'vlc', 'mpv', 'totem',  # Media players
                'libreoffice', 'okular', 'evince',  # Document viewers
            ]
            
            # Commands that typically take longer to execute
            long_running_commands = [
                'npm install', 'yarn install', 'pip install', 'apt install', 'apt-get install',
                'yum install', 'dnf install', 'pacman -S',  # Package installations
                'git clone', 'git pull', 'git push',  # Git operations
                'docker build', 'docker pull', 'docker run',  # Docker operations
                'make', 'cmake', 'gcc', 'g++', 'cargo build', 'mvn',  # Build commands
                'wget', 'curl', 'rsync', 'scp',  # Download/transfer commands
                'tar', 'unzip', 'gzip', 'bzip2',  # Compression commands
                'find /', 'grep -r',  # Recursive search operations
                'convert', 'ffmpeg', 'imagemagick',  # Media processing
            ]
            
            # Commands that are typically quick
            quick_commands = [
                'ls', 'pwd', 'cd', 'echo', 'cat', 'mkdir', 'rm', 'cp', 'mv',
                'touch', 'which', 'whereis', 'whoami', 'date', 'hostname',
                'uname', 'printenv', 'env', 'export', 'alias',
            ]
            
            # Check if command should run in background
            if background is None:
                # Auto-detect: check if command starts with a GUI app
                cmd_parts = command.split()
                if cmd_parts:
                    base_cmd = cmd_parts[0].split('/')[-1]  # Get command name without path
                    if base_cmd in gui_apps or command.endswith('&'):
                        background = True
                    else:
                        background = False
            
            # If background execution needed, use run_shell_async (no timeout constraint)
            if background:
                self.logger.info(f"Executing shell command in background: `{command}`")
                self.logger.warning(f"💡 Tip: For launching GUI applications, use 'open_application' tool instead for better control")
                # Remove trailing & if present since we're handling it
                command = command.rstrip('&').strip()
                return self.run_shell_async(command, timeout=0)
            
            # Intelligent timeout detection if not specified
            if timeout is None:
                cmd_lower = command.lower()
                
                # Check if it's a long-running command
                is_long_running = any(lr_cmd in cmd_lower for lr_cmd in long_running_commands)
                
                # Check if it's a quick command
                is_quick = any(cmd_lower.startswith(qc) for qc in quick_commands)
                
                if is_long_running:
                    timeout = 1800  # 30 minutes for long-running commands
                    self.logger.info(f"Detected long-running command, using {timeout}s timeout")
                elif is_quick:
                    timeout = 30  # 30 seconds for quick commands
                    self.logger.debug(f"Detected quick command, using {timeout}s timeout")
                else:
                    timeout = 300  # 5 minutes default
                    self.logger.debug(f"Using default {timeout}s timeout")
            
            self.logger.info(f"Executing shell command (timeout={timeout}s): `{command}`")

            # Enhanced security checks
            dangerous_patterns = [
                r"rm\s+-rf\s+/",  # Dangerous rm commands
                r"mkfs\.",  # Format commands
                r"dd\s+if=",  # Disk operations
                r">\s*/dev/",  # Redirecting to device files
                r"chmod\s+777",  # Dangerous permissions
                r"chown\s+-R\s+root",  # Dangerous ownership changes
                r"sudo\s+rm",  # Sudo with rm
                r"su\s+-",  # Switch user
                r"passwd",  # Password changes
                r"shutdown",  # System shutdown
                r"reboot",  # System reboot
                r"halt",  # System halt
                r"poweroff",  # Power off
                r"init\s+0",  # Init level 0
                r"killall",  # Kill all processes
                r"pkill\s+-9",  # Force kill
                r"kill\s+-9\s+-1",  # Kill all processes
            ]

            import re

            for pattern in dangerous_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return {
                        "success": False,
                        "error": f"Access denied: Potentially dangerous command detected: {command}",
                    }

            # Check for system directory access
            dangerous_dirs = [
                "/root",
                "/etc",
                "/sys",
                "/proc",
                "/dev",
                "/var",
                "/usr",
                "/bin",
                "/sbin",
                "/lib",
                "/lib64",
            ]
            for dangerous_dir in dangerous_dirs:
                if (
                    f"cd {dangerous_dir}" in command
                    or f"cd {dangerous_dir}/" in command
                ):
                    return {
                        "success": False,
                        "error": f"Access denied: System directory access not allowed: {command}",
                    }

            # Auto-quote file paths with spaces for Linux compatibility
            # Pattern to match file paths with spaces
            path_pattern = r"/[^\s]*\s[^\s]*"
            paths_with_spaces = re.findall(path_pattern, command)
            for path in paths_with_spaces:
                quoted_path = f'"{path}"'
                command = command.replace(path, quoted_path)

            # Log command execution (debug level only)
            self.logger.debug(f"🔧 Executing: {command}")

            # Execute command with enhanced error handling
            start_time = time.time()
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                timeout=timeout,
                cwd=settings.BASE_DIR,
                env=os.environ.copy(),  # Preserve environment
            )
            execution_time = time.time() - start_time

            # Capture output for return value with better error handling
            captured_result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=settings.BASE_DIR,
                env=os.environ.copy(),
            )

            # Log execution details
            self.logger.debug(f"Command executed in {execution_time:.2f} seconds")
            self.logger.debug(f"Return code: {captured_result.returncode}")

            # Format output properly
            output_parts = []
            if captured_result.stdout:
                output_parts.append(f"stdout:\n{captured_result.stdout}")
                self.logger.debug(f"📤 stdout:\n{captured_result.stdout}")
            if captured_result.stderr:
                output_parts.append(f"stderr:\n{captured_result.stderr}")
                self.logger.debug(f"⚠️ stderr:\n{captured_result.stderr}")

            output = "\n".join(output_parts) if output_parts else "No output"

            # Create comprehensive result
            result_dict = {
                "success": captured_result.returncode == 0,
                "output": output,
                "stdout": captured_result.stdout,
                "stderr": captured_result.stderr,
                "return_code": captured_result.returncode,
                "execution_time": execution_time,
                "command": command,
            }

            if captured_result.returncode == 0:
                self.logger.success(
                    f"Command completed successfully in {execution_time:.2f}s"
                )
            else:
                self.logger.error(
                    f"Command failed with return code {captured_result.returncode}"
                )
                result_dict["error"] = output

            return result_dict

        except subprocess.TimeoutExpired as e:
            error_msg = f"Command timed out after {timeout} seconds"
            self.logger.error(error_msg)
            self.logger.warning(f"💡 Tip: For long-running tasks, consider using run_shell_async() or increase timeout")
            return {"success": False, "error": error_msg, "command": command, "timeout": timeout}
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed with CalledProcessError: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "return_code": e.returncode,
                "command": command,
            }
        except FileNotFoundError as e:
            error_msg = f"Command not found: {e}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "command": command}
        except PermissionError as e:
            error_msg = f"Permission denied: {e}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "command": command}
        except Exception as e:
            error_msg = f"Unexpected error in command execution: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {"success": False, "error": error_msg, "command": command}

    def open_application(self, app_name: str, args: str = "") -> Dict[str, Any]:
        """
        Open/launch a GUI application in the background.
        This tool is specifically for launching applications that need to stay running.
        
        Args:
            app_name: Name of the application to launch (e.g., 'gedit', 'firefox', 'code')
            args: Additional arguments to pass to the application (e.g., filename for gedit)
            
        Returns:
            Dict with success status, process ID, and process info
            
        Examples:
            - open_application('gedit')
            - open_application('gedit', 'file.txt')
            - open_application('firefox', 'https://google.com')
            - open_application('code', '/path/to/project')
        """
        try:
            # Build the command
            command = f"{app_name} {args}".strip() if args else app_name
            
            self.logger.info(f"🚀 Launching application: {command}")
            
            # Always use run_shell_async with no timeout for GUI applications
            result = self.run_shell_async(command, timeout=0)
            
            if result.get("success"):
                self.logger.success(f"✅ Application '{app_name}' launched successfully")
                process_id = result.get("process_id")
                self.logger.info(f"📋 Process ID: {process_id}")
                self.logger.info("💡 Application is running in background. Use other tools to interact with it.")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to launch application '{app_name}': {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def run_shell_async(self, command: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Execute a shell command in a truly non-blocking separate process.
        Returns immediately and allows live interaction with the process.

        Args:
            command: The shell command to execute
            timeout: Maximum time to wait for command completion (0 = no timeout)

        Returns:
            Dict with success status, process info, and interaction methods
        """
        try:
            import subprocess
            import threading
            import signal
            import uuid
            import os

            # Validate input
            if not command or not command.strip():
                return {"success": False, "error": "Empty or invalid command provided"}

            command = command.strip()
            process_id = str(uuid.uuid4())[:8]
            self.logger.info(f"Starting async process {process_id}: `{command}`")

            # Enhanced security checks (same as run_shell)
            dangerous_patterns = [
                r"rm\s+-rf\s+/",  # Dangerous rm commands
                r"mkfs\.",  # Format commands
                r"dd\s+if=",  # Disk operations
                r">\s*/dev/",  # Redirecting to device files
                r"chmod\s+777",  # Dangerous permissions
                r"chown\s+-R\s+root",  # Dangerous ownership changes
                r"sudo\s+rm",  # Sudo with rm
                r"su\s+-",  # Switch user
                r"passwd",  # Password changes
                r"shutdown",  # System shutdown
                r"reboot",  # System reboot
                r"halt",  # System halt
                r"poweroff",  # Power off
                r"init\s+0",  # Init level 0
                r"killall",  # Kill all processes
                r"pkill\s+-9",  # Force kill
                r"kill\s+-9\s+-1",  # Kill all processes
            ]

            import re

            for pattern in dangerous_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return {
                        "success": False,
                        "error": f"Access denied: Potentially dangerous command detected: {command}",
                    }

            # Check for system directory access
            dangerous_dirs = [
                "/root",
                "/etc",
                "/sys",
                "/proc",
                "/dev",
                "/var",
                "/usr",
                "/bin",
                "/sbin",
                "/lib",
                "/lib64",
            ]
            for dangerous_dir in dangerous_dirs:
                if (
                    f"cd {dangerous_dir}" in command
                    or f"cd {dangerous_dir}/" in command
                ):
                    return {
                        "success": False,
                        "error": f"Access denied: System directory access not allowed: {command}",
                    }

            # Auto-quote file paths with spaces for Linux compatibility
            path_pattern = r"/[^\s]*\s[^\s]*"
            paths_with_spaces = re.findall(path_pattern, command)
            for path in paths_with_spaces:
                quoted_path = f'"{path}"'
                command = command.replace(path, quoted_path)

            # Start process in background
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                cwd=settings.BASE_DIR,
                env=os.environ.copy(),
                preexec_fn=os.setsid,  # Create new process group
                bufsize=0,  # Unbuffered for real-time interaction
            )

            # Store process info for later interaction
            process_info = {
                "pid": process.pid,
                "process_id": process_id,
                "command": command,
                "start_time": time.time(),
                "status": "running",
            }

            # Store in system context for later access
            if not hasattr(self.system, "running_processes"):
                self.system.running_processes = {}
            self.system.running_processes[process_id] = {
                "process": process,
                "info": process_info,
            }

            # Start monitoring thread if timeout is specified
            if timeout > 0:

                def monitor_process():
                    try:
                        process.wait(timeout=timeout)
                        process_info["status"] = "completed"
                        self.logger.info(f"Process {process_id} completed successfully")
                    except subprocess.TimeoutExpired:
                        process_info["status"] = "timeout"
                        try:
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            process.wait(timeout=5)
                        except (subprocess.TimeoutExpired, ProcessLookupError):
                            try:
                                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        self.logger.warning(
                            f"Process {process_id} timed out and was terminated"
                        )
                    except Exception as e:
                        process_info["status"] = "error"
                        self.logger.error(f"Process {process_id} error: {e}")

                monitor_thread = threading.Thread(target=monitor_process, daemon=True)
                monitor_thread.start()

            # Return immediately with process info
            return {
                "success": True,
                "output": f"Process {process_id} started successfully",
                "process_id": process_id,
                "pid": process.pid,
                "status": "running",
                "command": command,
                "message": f"Process started in background. Use process_id '{process_id}' for interaction.",
            }

        except Exception as e:
            return {"success": False, "error": f"Error starting async process: {e}"}

    def interact_with_process(
        self, process_id: str, action: str = "status", data: str = None
    ) -> Dict[str, Any]:
        """
        Interact with a running async process.

        Args:
            process_id: The process ID returned by run_shell_async
            action: Action to perform (status, kill, send_input, get_output)
            data: Data to send (for send_input action)

        Returns:
            Dict with interaction result
        """
        try:
            import signal
            import os

            if not hasattr(self.system, "running_processes"):
                return {"success": False, "error": "No running processes found"}

            if process_id not in self.system.running_processes:
                return {"success": False, "error": f"Process {process_id} not found"}

            process_data = self.system.running_processes[process_id]
            process = process_data["process"]
            info = process_data["info"]

            if action == "status":
                # Check if process is still running
                if process.poll() is None:
                    info["status"] = "running"
                    return {
                        "success": True,
                        "status": "running",
                        "pid": process.pid,
                        "uptime": time.time() - info["start_time"],
                        "command": info["command"],
                    }
                else:
                    info["status"] = "completed"
                    return {
                        "success": True,
                        "status": "completed",
                        "returncode": process.returncode,
                        "uptime": time.time() - info["start_time"],
                    }

            elif action == "kill":
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=5)
                    info["status"] = "killed"
                    return {
                        "success": True,
                        "message": f"Process {process_id} terminated gracefully",
                    }
                except (subprocess.TimeoutExpired, ProcessLookupError):
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        info["status"] = "force_killed"
                        return {
                            "success": True,
                            "message": f"Process {process_id} force killed",
                        }
                    except ProcessLookupError:
                        info["status"] = "already_dead"
                        return {
                            "success": True,
                            "message": f"Process {process_id} was already dead",
                        }

            elif action == "send_input":
                if data is None:
                    return {
                        "success": False,
                        "error": "No data provided for send_input",
                    }
                try:
                    process.stdin.write(data + "\n")
                    process.stdin.flush()
                    return {
                        "success": True,
                        "message": f"Sent input to process {process_id}",
                    }
                except Exception as e:
                    return {"success": False, "error": f"Failed to send input: {e}"}

            elif action == "get_output":
                try:
                    # Non-blocking read
                    import select

                    if select.select([process.stdout], [], [], 0.1)[0]:
                        output = process.stdout.read()
                        return {"success": True, "output": output}
                    else:
                        return {
                            "success": True,
                            "output": "",
                            "message": "No output available",
                        }
                except Exception as e:
                    return {"success": False, "error": f"Failed to read output: {e}"}

            else:
                return {"success": False, "error": f"Unknown action: {action}"}

        except Exception as e:
            return {"success": False, "error": f"Error interacting with process: {e}"}

    def install_package(self, package_name: str) -> Dict[str, Any]:
        """
        Install a Python package using pip in the current virtual environment.

        Args:
            package_name: The name of the package to install

        Returns:
            Dict with success status and output/error
        """
        import os

        venv_path = os.environ.get("VIRTUAL_ENV")
        if not venv_path:
            return {
                "success": False,
                "error": "No virtual environment detected. Please activate the venv first.",
                "output": "Virtual environment required for package installation",
            }

        self.logger.info(f"Installing package: {package_name} in venv")
        return self.run_shell(f"python3 -m pip install {package_name}")

    def read_file(
        self, file_path: str, encoding: str = "utf-8", max_size_mb: float = 100.0
    ) -> Dict[str, Any]:
        """
        Read the content of a file with robust error handling and validation.

        Args:
            file_path: The path to the file to read
            encoding: Text encoding to use (default: utf-8)
            max_size_mb: Maximum file size in MB to read (default: 100MB)

        Returns:
            Dict with success status and content/error
        """
        try:
            # Validate input
            if not file_path or not file_path.strip():
                return {
                    "success": False,
                    "error": "Empty or invalid file path provided",
                }

            file_path = file_path.strip()
            self.logger.debug(f"Reading file: {file_path}")

            # Resolve path
            path = self._resolve_path(file_path)

            # Check if file exists
            if not path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            # Check if it's a file (not directory)
            if not path.is_file():
                return {"success": False, "error": f"Path is not a file: {file_path}"}

            # Check file size
            file_size = path.stat().st_size
            max_size_bytes = max_size_mb * 1024 * 1024
            if file_size > max_size_bytes:
                return {
                    "success": False,
                    "error": f"File too large: {file_size / (1024*1024):.1f}MB (max {max_size_mb}MB)",
                }

            # Check if file is readable
            if not os.access(path, os.R_OK):
                return {"success": False, "error": f"File not readable: {file_path}"}

            self.logger.debug(f"File size: {file_size / 1024:.1f}KB")

            # Try to read file with different encodings if needed
            encodings_to_try = [encoding, "utf-8", "latin-1", "cp1252", "iso-8859-1"]
            content = None
            used_encoding = None

            for enc in encodings_to_try:
                try:
                    with open(path, "r", encoding=enc) as f:
                        content = f.read()
                    used_encoding = enc
                    self.logger.debug(f"Successfully read file with encoding: {enc}")
                    break
                except UnicodeDecodeError:
                    self.logger.debug(
                        f"Failed to read with encoding {enc}, trying next..."
                    )
                    continue
                except Exception as e:
                    self.logger.debug(f"Error reading with encoding {enc}: {e}")
                    continue

            if content is None:
                return {
                    "success": False,
                    "error": f"Failed to read file with any supported encoding: {file_path}",
                }

            # Validate content
            if not content:
                self.logger.warning(f"File is empty: {file_path}")

            result = {
                "success": True,
                "output": content,
                "file_path": str(path),
                "file_size": file_size,
                "encoding": used_encoding,
                "lines": len(content.splitlines()) if content else 0,
                "characters": len(content) if content else 0,
            }

            self.logger.success(
                f"Successfully read file: {file_path} ({file_size / 1024:.1f}KB)"
            )
            return result

        except PermissionError as e:
            return {"success": False, "error": f"Permission denied reading file: {e}"}
        except OSError as e:
            return {"success": False, "error": f"OS error reading file: {e}"}
        except Exception as e:
            self.logger.error(f"Unexpected error reading file: {e}")
            return {"success": False, "error": f"Failed to read file: {str(e)}"}

    def write_file(
        self,
        file_path: str,
        content: str,
        encoding: str = "utf-8",
        create_backup: bool = False,
    ) -> Dict[str, Any]:
        """
        Write content to a file with robust error handling and validation.

        Args:
            file_path: The path to the file to write
            content: The content to write
            encoding: Text encoding to use (default: utf-8)
            create_backup: Whether to create a backup if file exists (default: False)

        Returns:
            Dict with success status and message/error
        """
        try:
            # Validate inputs
            if not file_path or not file_path.strip():
                return {
                    "success": False,
                    "error": "Empty or invalid file path provided",
                }

            if content is None:
                content = ""

            file_path = file_path.strip()
            self.logger.debug(f"Writing file: {file_path}")

            # Resolve path
            path = self._resolve_path(file_path)

            # Check if file already exists and create backup if requested
            if path.exists() and create_backup:
                backup_path = path.with_suffix(path.suffix + ".backup")
                try:
                    import shutil

                    shutil.copy2(path, backup_path)
                    self.logger.debug(f"Created backup: {backup_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to create backup: {e}")

            # Check if parent directory exists and create if needed
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                return {
                    "success": False,
                    "error": f"Permission denied creating directory: {e}",
                }
            except Exception as e:
                return {"success": False, "error": f"Failed to create directory: {e}"}

            # Check if we can write to the location
            if path.exists() and not os.access(path, os.W_OK):
                return {"success": False, "error": f"File not writable: {file_path}"}

            # Check if parent directory is writable
            if not os.access(path.parent, os.W_OK):
                return {
                    "success": False,
                    "error": f"Directory not writable: {path.parent}",
                }

            # Write content with proper error handling
            try:
                with open(path, "w", encoding=encoding) as f:
                    f.write(content)
            except UnicodeEncodeError as e:
                # Try with different encoding if the specified one fails
                self.logger.warning(f"Failed to write with {encoding}, trying utf-8...")
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                    encoding = "utf-8"
                except Exception as e2:
                    return {
                        "success": False,
                        "error": f"Failed to write file with any encoding: {e2}",
                    }
            except Exception as e:
                return {"success": False, "error": f"Failed to write file: {e}"}

            # Verify the file was written correctly
            if not path.exists():
                return {
                    "success": False,
                    "error": "File was not created after write operation",
                }

            # Get file stats
            file_size = path.stat().st_size
            lines = len(content.splitlines()) if content else 0
            characters = len(content) if content else 0

            result = {
                "success": True,
                "output": f"Successfully wrote content to {path}",
                "file_path": str(path),
                "file_size": file_size,
                "encoding": encoding,
                "lines": lines,
                "characters": characters,
                "backup_created": create_backup and path.exists(),
            }

            self.logger.success(
                f"Successfully wrote file: {file_path} ({file_size / 1024:.1f}KB)"
            )
            return result

        except PermissionError as e:
            return {"success": False, "error": f"Permission denied writing file: {e}"}
        except OSError as e:
            return {"success": False, "error": f"OS error writing file: {e}"}
        except Exception as e:
            self.logger.error(f"Unexpected error writing file: {e}")
            return {"success": False, "error": f"Failed to write file: {str(e)}"}

    def list_dir(self, directory: str = ".") -> Dict[str, Any]:
        """
        List the contents of a directory with robust error handling and detailed information.

        Args:
            directory: The path to the directory to list

        Returns:
            Dict with success status and directory contents/error
        """
        try:
            # Validate input
            if not directory:
                directory = "."

            self.logger.debug(f"Listing directory: {directory}")

            # Resolve the path
            path = self._resolve_path(directory)

            # Check if path exists
            if not path.exists():
                return {"success": False, "error": f"Directory not found: {directory}"}

            # Check if it's a directory
            if not path.is_dir():
                return {
                    "success": False,
                    "error": f"Path is not a directory: {directory}",
                }

            # Check if directory is readable
            if not os.access(path, os.R_OK):
                return {
                    "success": False,
                    "error": f"Directory not readable: {directory}",
                }

            # Get directory contents with detailed information
            try:
                items = []
                files = []
                directories = []

                for item in path.iterdir():
                    try:
                        item_path = str(item)
                        item_stat = item.stat()

                        # Determine if it's a file or directory
                        is_file = item.is_file()
                        is_dir = item.is_dir()
                        is_symlink = item.is_symlink()

                        # Get file size
                        size = item_stat.st_size if is_file else 0

                        # Format size
                        if size < 1024:
                            size_str = f"{size}B"
                        elif size < 1024 * 1024:
                            size_str = f"{size / 1024:.1f}KB"
                        else:
                            size_str = f"{size / (1024 * 1024):.1f}MB"

                        # Create item info
                        item_info = {
                            "name": item.name,
                            "path": item_path,
                            "is_file": is_file,
                            "is_directory": is_dir,
                            "is_symlink": is_symlink,
                            "size": size,
                            "size_str": size_str,
                            "modified": item_stat.st_mtime,
                        }

                        items.append(item_info)

                        if is_file:
                            files.append(item_info)
                        elif is_dir:
                            directories.append(item_info)

                    except (OSError, PermissionError) as e:
                        # Skip items we can't access
                        self.logger.debug(f"Cannot access {item}: {e}")
                        continue

                # Sort items: directories first, then files, both alphabetically
                directories.sort(key=lambda x: x["name"].lower())
                files.sort(key=lambda x: x["name"].lower())
                sorted_items = directories + files

                # Create output string
                output_lines = []
                for item in sorted_items:
                    if item["is_directory"]:
                        output_lines.append(f"{item['path']}/")
                    else:
                        output_lines.append(item["path"])

                output = "\n".join(output_lines)

                # Create detailed result
                result = {
                    "success": True,
                    "output": output,
                    "directory": str(path),
                    "total_items": len(items),
                    "files": len(files),
                    "directories": len(directories),
                    "items": sorted_items,
                    "files_list": [item["name"] for item in files],
                    "directories_list": [item["name"] for item in directories],
                }

                self.logger.success(
                    f"Successfully listed directory: {directory} ({len(items)} items)"
                )
                return result

            except PermissionError as e:
                return {
                    "success": False,
                    "error": f"Permission denied listing directory: {e}",
                }
            except OSError as e:
                return {"success": False, "error": f"OS error listing directory: {e}"}

        except Exception as e:
            self.logger.error(f"Unexpected error listing directory: {e}")
            return {"success": False, "error": f"Failed to list directory: {str(e)}"}

    def change_dir(self, directory: str) -> Dict[str, Any]:
        """
        Change the current working directory.

        Args:
            directory: The path to the directory to change to

        Returns:
            Dict with success status and current directory/error
        """
        try:
            # Resolve the path and check if it's within allowed directories
            path = Path(directory).resolve()
            base_dir = Path(settings.BASE_DIR).resolve()

            # Allow access to project directory, user's home directory, and current directory
            # Only block critical system directories
            if not (
                path.is_relative_to(base_dir)
                or path.is_relative_to(Path.home())
                or str(path) == "."
                or str(path) == ".."
            ):
                # Check if it's a relative path that should be allowed
                if not (
                    str(directory).startswith(".") or str(directory).startswith("~")
                ):
                    return {
                        "success": False,
                        "error": f"Access denied: Directory outside allowed scope: {directory}",
                    }

            # Block access to critical system directories only
            blocked_dirs = [
                "/etc",
                "/sys",
                "/proc",
                "/dev",
                "/var",
                "/usr",
                "/root",
                "/bin",
                "/sbin",
                "/lib",
                "/lib64",
            ]
            if any(str(path).startswith(blocked_dir) for blocked_dir in blocked_dirs):
                return {
                    "success": False,
                    "error": f"Access denied: System directory access not allowed: {directory}",
                }

            # Block access to root directory only
            if str(path) == "/":
                return {
                    "success": False,
                    "error": f"Access denied: Root directory access not allowed: {directory}",
                }

            if not path.exists():
                return {"success": False, "error": f"Directory not found: {directory}"}

            if not path.is_dir():
                return {
                    "success": False,
                    "error": f"Path is not a directory: {directory}",
                }

            # Change directory
            os.chdir(path)

            # Update context manager with new directory
            if hasattr(self.system, "context_manager"):
                from src.core.context_manager import ContextType, Priority, ContextRelevance
                self.system.context_manager.add_context_entry(
                    ContextType.SYSTEM_EVENT,
                    f"Changed directory to: {path}",
                    Priority.NORMAL,
                    ContextRelevance.CONTEXTUAL,
                    metadata={"new_directory": str(path)},
                )

            return {
                "success": True,
                "output": f"Changed directory to: {path}",
                "current_dir": str(path),
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to change directory: {e}"}

    def create_and_save_tool(
        self, tool_name: str, tool_code: str, doc_string: str
    ) -> Dict[str, Any]:
        """
        Create and save a new dynamic tool.

        Args:
            tool_name: The name of the tool/function to create
            tool_code: The Python code for the function
            doc_string: A description of the tool's purpose

        Returns:
            Dict with success status and message/error
        """
        self.logger.info(f"Creating and saving tool '{tool_name}'...")

        try:
            # Validate syntax
            ast.parse(tool_code)

            # Create execution context with common modules
            import os
            import json
            import random
            import sys
            import time
            import datetime

            exec_context = {
                "system": self.system,
                "os": os,
                "json": json,
                "random": random,
                "sys": sys,
                "time": time,
                "datetime": datetime,
            }
            exec(tool_code, globals(), exec_context)
            func = exec_context.get(tool_name)

            if not func:
                return {
                    "success": False,
                    "error": f"Function '{tool_name}' not found in the provided code.",
                }

            # Create tool object
            from database.memory import Tool

            new_tool = Tool(
                name=tool_name,
                code=tool_code,
                doc=doc_string,
                is_dynamic=True,
                last_used=time.time(),
                func=func,
            )

            # Register with system if available
            if self.system and hasattr(self.system, "tool_manager"):
                self.system.tool_manager.register_tool(new_tool)

            # Save to file
            tool_path = settings.TOOLS_DIR / f"{tool_name}.py"
            with open(tool_path, "w", encoding="utf-8") as f:
                f.write(tool_code)

            return {
                "success": True,
                "output": f"Tool '{tool_name}' created and registered successfully.",
            }

        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error in tool code: {e}"}
        except Exception as e:
            error_msg = f"Error creating and saving tool: {e}\n{traceback.format_exc()}"
            return {"success": False, "error": error_msg}

    def complete_task(self, message: str) -> Dict[str, Any]:
        """
        Indicate that a task is complete.

        Args:
            message: A summary of the completed task

        Returns:
            Dict with success status and completion signal
        """
        self.logger.success(f"Task Complete: {message}")
        return {"success": True, "output": "TASK_COMPLETED_SIGNAL"}

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get current system information.

        Returns:
            Dict with system information
        """
        try:
            os_info = self.run_shell("uname -a")
            python_version = self.run_shell("python3 --version")

            return {
                "success": True,
                "output": {
                    "os_info": os_info.get("output", "Unknown"),
                    "python_version": python_version.get("output", "Unknown"),
                    "platform": sys.platform,
                    "working_directory": str(settings.BASE_DIR),
                },
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get system info: {e}"}

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get detailed information about a file."""
        try:
            resolved_path = self._resolve_path(file_path)
            if not resolved_path.exists():
                return {"success": False, "error": f"File not found: {resolved_path}"}

            stat = resolved_path.stat()
            return {
                "success": True,
                "output": {
                    "name": resolved_path.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "is_file": resolved_path.is_file(),
                    "is_dir": resolved_path.is_dir(),
                    "extension": resolved_path.suffix,
                    "parent": str(resolved_path.parent),
                },
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get file info: {e}"}

    def copy_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Copy a file from source to destination."""
        try:
            source_path = self._resolve_path(source)
            dest_path = self._resolve_path(destination)

            if not source_path.exists():
                return {
                    "success": False,
                    "error": f"Source file not found: {source_path}",
                }

            # Create destination directory if it doesn't exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            import shutil

            shutil.copy2(source_path, dest_path)
            return {
                "success": True,
                "output": f"File copied from {source_path} to {dest_path}",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to copy file: {e}"}

    def move_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Move a file from source to destination."""
        try:
            source_path = self._resolve_path(source)
            dest_path = self._resolve_path(destination)

            if not source_path.exists():
                return {
                    "success": False,
                    "error": f"Source file not found: {source_path}",
                }

            # Create destination directory if it doesn't exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            import shutil

            shutil.move(str(source_path), str(dest_path))
            return {
                "success": True,
                "output": f"File moved from {source_path} to {dest_path}",
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to move file: {e}"}

    def delete_file(self, file_path: str) -> Dict[str, Any]:
        """Delete a file or directory."""
        try:
            resolved_path = self._resolve_path(file_path)
            if not resolved_path.exists():
                return {"success": False, "error": f"File not found: {resolved_path}"}

            if resolved_path.is_file():
                resolved_path.unlink()
                return {"success": True, "output": f"File deleted: {resolved_path}"}
            elif resolved_path.is_dir():
                import shutil

                shutil.rmtree(resolved_path)
                return {
                    "success": True,
                    "output": f"Directory deleted: {resolved_path}",
                }
            else:
                return {
                    "success": False,
                    "error": f"Unknown file type: {resolved_path}",
                }
        except Exception as e:
            return {"success": False, "error": f"Failed to delete file: {e}"}

    def create_directory(self, directory_path: str) -> Dict[str, Any]:
        """Create a directory."""
        try:
            resolved_path = self._resolve_path(directory_path)
            resolved_path.mkdir(parents=True, exist_ok=True)
            return {"success": True, "output": f"Directory created: {resolved_path}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to create directory: {e}"}

    def get_directory_size(self, directory_path: str) -> Dict[str, Any]:
        """Get the total size of a directory."""
        try:
            resolved_path = self._resolve_path(directory_path)
            if not resolved_path.is_dir():
                return {"success": False, "error": f"Not a directory: {resolved_path}"}

            total_size = 0
            file_count = 0
            dir_count = 0

            for item in resolved_path.rglob("*"):
                if item.is_file():
                    total_size += item.stat().st_size
                    file_count += 1
                elif item.is_dir():
                    dir_count += 1

            return {
                "success": True,
                "output": {
                    "total_size": total_size,
                    "file_count": file_count,
                    "dir_count": dir_count,
                    "size_mb": round(total_size / 1024 / 1024, 2),
                },
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get directory size: {e}"}

    def find_large_files(
        self, directory: str, min_size_mb: float = 10.0
    ) -> Dict[str, Any]:
        """Find files larger than specified size in MB."""
        try:
            resolved_path = self._resolve_path(directory)
            if not resolved_path.is_dir():
                return {"success": False, "error": f"Not a directory: {resolved_path}"}

            min_size_bytes = min_size_mb * 1024 * 1024
            large_files = []

            for item in resolved_path.rglob("*"):
                if item.is_file():
                    size = item.stat().st_size
                    if size >= min_size_bytes:
                        large_files.append(
                            {"path": str(item), "size_mb": round(size / 1024 / 1024, 2)}
                        )

            return {
                "success": True,
                "output": {
                    "large_files": large_files,
                    "count": len(large_files),
                    "min_size_mb": min_size_mb,
                },
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to find large files: {e}"}

    def get_system_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage information for all mounted drives."""
        try:
            import shutil

            disk_usage = shutil.disk_usage("/")

            return {
                "success": True,
                "output": {
                    "total_gb": round(disk_usage.total / 1024 / 1024 / 1024, 2),
                    "used_gb": round(disk_usage.used / 1024 / 1024 / 1024, 2),
                    "free_gb": round(disk_usage.free / 1024 / 1024 / 1024, 2),
                    "usage_percent": round(
                        (disk_usage.used / disk_usage.total) * 100, 2
                    ),
                },
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get disk usage: {e}"}

    def get_process_info(self) -> Dict[str, Any]:
        """Get information about running processes."""
        try:
            import psutil

            processes = []
            for proc in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_percent"]
            ):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            return {
                "success": True,
                "output": {
                    "processes": processes[:20],  # Limit to first 20 processes
                    "total_processes": len(processes),
                },
            }
        except ImportError:
            return {"success": False, "error": "psutil not available"}
        except Exception as e:
            return {"success": False, "error": f"Failed to get process info: {e}"}

    def navigate_to_user_directories(self) -> Dict[str, Any]:
        """Navigate to common user directories based on OS."""
        try:
            import os
            import sys

            # Get current username
            username = os.getenv("USER") or os.getenv("USERNAME") or "user"

            # Dynamically find common user directories
            if sys.platform == "win32":
                base_path = f"C:\\Users\\{username}"
            elif sys.platform == "darwin":  # macOS
                base_path = f"/Users/{username}"
            else:  # Linux
                base_path = f"/home/{username}"
            
            # Try to find common directories dynamically
            root_paths = []
            common_dirs = ["Desktop", "Documents", "Pictures", "Downloads", "Projects"]
            
            for dir_name in common_dirs:
                potential_path = os.path.join(base_path, dir_name)
                if os.path.exists(potential_path):
                    root_paths.append(potential_path)

            # Try each path
            for path in root_paths:
                if os.path.exists(path):
                    # Try to navigate to it
                    change_result = self.change_dir(path)
                    if change_result.get("success"):
                        return {
                            "success": True,
                            "output": f"Successfully navigated to: {path}",
                            "path": path,
                        }

            return {"success": False, "error": "No common user directories found"}
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to navigate to user directories: {e}",
            }

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path to an absolute Path object."""
        if not path:
            return Path.cwd()

        # Handle relative paths
        if not os.path.isabs(path):
            # If it's a relative path, resolve it relative to current working directory
            return Path.cwd() / path
        else:
            # It's already an absolute path
            return Path(path)

    def search_in_file(
        self, file_path: str, search_term: str, case_sensitive: bool = False
    ) -> Dict[str, Any]:
        """Search for text in a file."""
        try:
            resolved_path = self._resolve_path(file_path)
            if not resolved_path.is_file():
                return {"success": False, "error": f"File not found: {resolved_path}"}

            content = resolved_path.read_text(encoding="utf-8")
            matches = []

            if not case_sensitive:
                content_lower = content.lower()
                search_lower = search_term.lower()
                start = 0
                while True:
                    pos = content_lower.find(search_lower, start)
                    if pos == -1:
                        break
                    matches.append(
                        {"position": pos, "line": content[:pos].count("\n") + 1}
                    )
                    start = pos + 1
            else:
                start = 0
                while True:
                    pos = content.find(search_term, start)
                    if pos == -1:
                        break
                    matches.append(
                        {"position": pos, "line": content[:pos].count("\n") + 1}
                    )
                    start = pos + 1

            return {"success": True, "matches": matches, "match_count": len(matches)}
        except Exception as e:
            return {"success": False, "error": f"Error searching in file: {e}"}

    def replace_in_file(
        self, file_path: str, old_text: str, new_text: str, case_sensitive: bool = True
    ) -> Dict[str, Any]:
        """Replace text in a file."""
        try:
            resolved_path = self._resolve_path(file_path)
            if not resolved_path.is_file():
                return {"success": False, "error": f"File not found: {resolved_path}"}

            content = resolved_path.read_text(encoding="utf-8")

            if case_sensitive:
                new_content = content.replace(old_text, new_text)
            else:
                import re

                new_content = re.sub(old_text, new_text, content, flags=re.IGNORECASE)

            if new_content == content:
                return {
                    "success": True,
                    "message": "No replacements made.",
                    "replacements": 0,
                }

            replacements = (
                content.count(old_text)
                if case_sensitive
                else len(re.findall(old_text, content, flags=re.IGNORECASE))
            )

            resolved_path.write_text(new_content, encoding="utf-8")
            return {
                "success": True,
                "message": f"Replaced {replacements} occurrences.",
                "replacements": replacements,
            }
        except Exception as e:
            return {"success": False, "error": f"Error replacing in file: {e}"}

    def search_directory(
        self, directory: str, search_term: str, case_sensitive: bool = False
    ) -> Dict[str, Any]:
        """Recursively search for a term in all files within a directory."""
        try:
            resolved_path = self._resolve_path(directory)
            if not resolved_path.is_dir():
                return {
                    "success": False,
                    "error": f"Directory not found: {resolved_path}",
                }

            matches = []
            for root, _, files in os.walk(resolved_path):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        with open(
                            file_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            content = f.read()

                        if not case_sensitive:
                            if search_term.lower() in content.lower():
                                matches.append(
                                    str(file_path.relative_to(resolved_path))
                                )
                        else:
                            if search_term in content:
                                matches.append(
                                    str(file_path.relative_to(resolved_path))
                                )
                    except Exception:
                        continue  # Ignore files that can't be read

            return {"success": True, "matches": matches, "match_count": len(matches)}
        except Exception as e:
            return {"success": False, "error": f"Error searching directory: {e}"}

    def find_files(self, pattern: str, directory: str = ".") -> Dict[str, Any]:
        """Find files matching a glob pattern."""
        try:
            resolved_path = self._resolve_path(directory)
            if not resolved_path.is_dir():
                return {
                    "success": False,
                    "error": f"Directory not found: {resolved_path}",
                }

            found_files = [
                str(p.relative_to(resolved_path)) for p in resolved_path.glob(pattern)
            ]
            return {"success": True, "files": found_files, "count": len(found_files)}
        except Exception as e:
            return {"success": False, "error": f"Error finding files: {e}"}

    def create_archive(
        self, archive_path: str, files: list, archive_type: str = "zip"
    ) -> Dict[str, Any]:
        """Create a zip or tar archive from a list of files."""
        try:
            import zipfile
            import tarfile

            resolved_archive_path = self._resolve_path(archive_path)
            resolved_archive_path.parent.mkdir(parents=True, exist_ok=True)

            resolved_files = [
                self._resolve_path(f) for f in files if self._resolve_path(f).exists()
            ]
            if not resolved_files:
                return {"success": False, "error": "No valid files found to archive."}

            if archive_type.lower() == "zip":
                with zipfile.ZipFile(
                    resolved_archive_path, "w", zipfile.ZIP_DEFLATED
                ) as zipf:
                    for file_path in resolved_files:
                        zipf.write(file_path, file_path.name)
            elif archive_type.lower() == "tar":
                with tarfile.open(resolved_archive_path, "w") as tarf:
                    for file_path in resolved_files:
                        tarf.add(file_path, arcname=file_path.name)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported archive type: {archive_type}",
                }

            return {
                "success": True,
                "message": f"Archive created: {resolved_archive_path}",
            }
        except Exception as e:
            return {"success": False, "error": f"Error creating archive: {e}"}

    def extract_archive(
        self, archive_path: str, extract_to: str = None
    ) -> Dict[str, Any]:
        """Extract a zip or tar archive."""
        try:
            import zipfile
            import tarfile

            resolved_archive_path = self._resolve_path(archive_path)
            if not resolved_archive_path.is_file():
                return {
                    "success": False,
                    "error": f"Archive not found: {resolved_archive_path}",
                }

            extract_dir = (
                self._resolve_path(extract_to)
                if extract_to
                else resolved_archive_path.parent / resolved_archive_path.stem
            )
            extract_dir.mkdir(parents=True, exist_ok=True)

            if resolved_archive_path.suffix == ".zip":
                with zipfile.ZipFile(resolved_archive_path, "r") as zipf:
                    zipf.extractall(extract_dir)
            elif ".tar" in resolved_archive_path.suffixes:
                with tarfile.open(resolved_archive_path, "r:*") as tarf:
                    tarf.extractall(extract_dir)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported archive format: {resolved_archive_path.name}",
                }

            return {"success": True, "message": f"Archive extracted to: {extract_dir}"}
        except Exception as e:
            return {"success": False, "error": f"Error extracting archive: {e}"}

    def read_json_file(self, file_path: str) -> Dict[str, Any]:
        """Read and parse a JSON file."""
        try:
            resolved_path = self._resolve_path(file_path)
            if not resolved_path.is_file():
                return {"success": False, "error": f"File not found: {resolved_path}"}

            with open(resolved_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return {"success": True, "data": data}
        except Exception as e:
            return {"success": False, "error": f"Error reading JSON file: {e}"}

    def write_json_file(
        self, file_path: str, data: Any, indent: int = 2
    ) -> Dict[str, Any]:
        """Write data to a JSON file."""
        try:
            resolved_path = self._resolve_path(file_path)
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent)
            return {"success": True, "message": f"JSON file written: {resolved_path}"}
        except Exception as e:
            return {"success": False, "error": f"Error writing JSON file: {e}"}

    def read_csv_file(self, file_path: str, delimiter: str = ",") -> Dict[str, Any]:
        """Read and parse a CSV file."""
        try:
            import csv

            resolved_path = self._resolve_path(file_path)
            if not resolved_path.is_file():
                return {"success": False, "error": f"File not found: {resolved_path}"}

            with open(resolved_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=delimiter)
                rows = list(reader)

            return {"success": True, "rows": rows}
        except Exception as e:
            return {"success": False, "error": f"Error reading CSV file: {e}"}

    def write_csv_file(
        self, file_path: str, data: list, headers: list = None, delimiter: str = ","
    ) -> Dict[str, Any]:
        """Write data to a CSV file."""
        try:
            import csv

            resolved_path = self._resolve_path(file_path)
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter=delimiter)
                if headers:
                    writer.writerow(headers)
                writer.writerows(data)
            return {"success": True, "message": f"CSV file written: {resolved_path}"}
        except Exception as e:
            return {"success": False, "error": f"Error writing CSV file: {e}"}

    def run_linter(self, path: str) -> Dict[str, Any]:
        """Run a Python linter (pylint) on a file or directory."""
        try:
            import pylint.lint
        except ImportError:
            self.logger.info("Pylint not found, attempting to install in venv.")
            install_result = self.install_package("pylint")
            if not install_result["success"]:
                return {
                    "success": False,
                    "error": "Pylint is not installed and installation failed.",
                }
            # After successful installation, try to import again
            try:
                import importlib

                importlib.invalidate_caches()  # Invalidate cache to ensure fresh import
                import pylint.lint
            except ImportError as e:
                return {
                    "success": False,
                    "error": f"Pylint installed but import failed: {e}",
                }

        resolved_path = self._resolve_path(path)

        # Capture stdout and stderr using subprocess instead of pylint's Run class
        try:
            import subprocess

            result = subprocess.run(
                ["pylint", str(resolved_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "output": (
                    f"Linter output:\n{result.stdout}\n{result.stderr}"
                    if result.stdout or result.stderr
                    else "No linting issues found"
                ),
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Pylint execution timed out"}
        except Exception as e:
            return {"success": False, "error": f"Error running pylint: {e}"}

    def replace_in_multiple_files(
        self, files: list, old_text: str, new_text: str, case_sensitive: bool = True
    ) -> Dict[str, Any]:
        """Perform a search and replace operation across multiple files."""
        results = []
        for file_path in files:
            result = self.replace_in_file(file_path, old_text, new_text, case_sensitive)
            results.append({"file_path": file_path, "result": result})

        return {"success": True, "results": results}

    def scroll_screen(
        self, x: int, y: int, direction: str = "up", amount: int = 3
    ) -> Dict[str, Any]:
        """Scroll at specific screen coordinates using xdotool."""
        try:
            import subprocess

            # Check if xdotool is available
            check_result = self.check_system_dependency("xdotool")
            if not check_result.get("installed", False):
                return {
                    "success": False,
                    "error": "xdotool not installed",
                    "output": f"xdotool is required for screen scrolling. {check_result.get('instructions', 'Please install xdotool manually.')}",
                    "instructions": check_result.get(
                        "instructions", "Please install xdotool manually."
                    ),
                }

            # Move mouse to position first
            move_cmd = ["xdotool", "mousemove", str(x), str(y)]
            move_result = subprocess.run(
                move_cmd, capture_output=True, text=True, timeout=5
            )

            if move_result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Failed to move mouse: {move_result.stderr}",
                    "output": f"Could not move mouse to ({x}, {y})",
                }

            # Perform scroll
            scroll_cmd = [
                "xdotool",
                "click",
                "--clearmodifiers",
                "--repeat",
                str(amount),
            ]
            if direction == "up":
                scroll_cmd.extend(["4"])  # Scroll up
            elif direction == "down":
                scroll_cmd.extend(["5"])  # Scroll down
            elif direction == "left":
                scroll_cmd.extend(["6"])  # Scroll left
            elif direction == "right":
                scroll_cmd.extend(["7"])  # Scroll right
            else:
                return {
                    "success": False,
                    "error": f"Invalid scroll direction: {direction}",
                    "output": "Valid directions: up, down, left, right",
                }

            result = subprocess.run(
                scroll_cmd, capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": f"Scrolled {direction} {amount} times at ({x}, {y})",
                    "message": f"Successfully scrolled {direction} at ({x}, {y})",
                }
            else:
                return {
                    "success": False,
                    "error": f"xdotool scroll failed: {result.stderr}",
                    "output": f"Failed to scroll {direction} at ({x}, {y})",
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Scroll operation timed out"}
        except Exception as e:
            return {"success": False, "error": f"Error scrolling screen: {e}"}

    def move_mouse(self, x: int, y: int) -> Dict[str, Any]:
        """Move mouse to specific coordinates using xdotool."""
        try:
            import subprocess

            # Check if xdotool is available
            check_result = self.check_system_dependency("xdotool")
            if not check_result.get("installed", False):
                return {
                    "success": False,
                    "error": "xdotool not installed",
                    "output": f"xdotool is required for mouse movement. {check_result.get('instructions', 'Please install xdotool manually.')}",
                    "instructions": check_result.get(
                        "instructions", "Please install xdotool manually."
                    ),
                }

            cmd = ["xdotool", "mousemove", str(x), str(y)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": f"Mouse moved to ({x}, {y})",
                    "message": f"Successfully moved mouse to ({x}, {y})",
                }
            else:
                return {
                    "success": False,
                    "error": f"xdotool failed: {result.stderr}",
                    "output": f"Failed to move mouse to ({x}, {y})",
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Mouse movement timed out"}
        except Exception as e:
            return {"success": False, "error": f"Error moving mouse: {e}"}

    def drag_mouse(
        self, x1: int, y1: int, x2: int, y2: int, duration: float = 1.0
    ) -> Dict[str, Any]:
        """Drag mouse from one point to another using xdotool."""
        try:
            import subprocess

            # Check if xdotool is available
            check_result = self.check_system_dependency("xdotool")
            if not check_result.get("installed", False):
                return {
                    "success": False,
                    "error": "xdotool not installed",
                    "output": f"xdotool is required for mouse dragging. {check_result.get('instructions', 'Please install xdotool manually.')}",
                    "instructions": check_result.get(
                        "instructions", "Please install xdotool manually."
                    ),
                }

            # Move to start position, press button 1, drag to end position, release
            cmd = [
                "xdotool",
                "mousemove",
                str(x1),
                str(y1),
                "mousedown",
                "1",
                "mousemove",
                "--sync",
                str(x2),
                str(y2),
                "mouseup",
                "1",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": f"Dragged from ({x1}, {y1}) to ({x2}, {y2})",
                    "message": f"Successfully dragged mouse from ({x1}, {y1}) to ({x2}, {y2})",
                }
            else:
                return {
                    "success": False,
                    "error": f"xdotool drag failed: {result.stderr}",
                    "output": f"Failed to drag from ({x1}, {y1}) to ({x2}, {y2})",
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Drag operation timed out"}
        except Exception as e:
            return {"success": False, "error": f"Error dragging mouse: {e}"}

    def type_text(self, text: str, x: int = None, y: int = None) -> Dict[str, Any]:
        """Type text at specific coordinates or current position using xdotool."""
        try:
            import subprocess

            # Check if xdotool is available
            check_result = self.check_system_dependency("xdotool")
            if not check_result.get("installed", False):
                return {
                    "success": False,
                    "error": "xdotool not installed",
                    "output": f"xdotool is required for text input. {check_result.get('instructions', 'Please install xdotool manually.')}",
                    "instructions": check_result.get(
                        "instructions", "Please install xdotool manually."
                    ),
                }

            cmd = ["xdotool", "type", "--clearmodifiers", text]

            # If coordinates provided, click there first
            if x is not None and y is not None:
                # First move mouse to coordinates, then click
                move_cmd = ["xdotool", "mousemove", str(x), str(y)]
                move_result = subprocess.run(
                    move_cmd, capture_output=True, text=True, timeout=3
                )
                if move_result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Failed to move mouse to coordinates: {move_result.stderr}",
                        "output": f"Could not move mouse to ({x}, {y})",
                    }
                
                # Then click at the current position
                click_cmd = ["xdotool", "click", "1"]
                click_result = subprocess.run(
                    click_cmd, capture_output=True, text=True, timeout=3
                )
                if click_result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Failed to click before typing: {click_result.stderr}",
                        "output": f"Could not click at ({x}, {y}) before typing",
                    }

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                location = (
                    f" at ({x}, {y})"
                    if x is not None and y is not None
                    else " at current position"
                )
                return {
                    "success": True,
                    "output": f"Typed '{text}'{location}",
                    "message": f"Successfully typed text{location}",
                }
            else:
                return {
                    "success": False,
                    "error": f"xdotool type failed: {result.stderr}",
                    "output": f"Failed to type text",
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Text input timed out"}
        except Exception as e:
            return {"success": False, "error": f"Error typing text: {e}"}

    def press_key(self, key: str) -> Dict[str, Any]:
        """Press key combination using xdotool."""
        try:
            import subprocess

            # Check if xdotool is available
            check_result = self.check_system_dependency("xdotool")
            if not check_result.get("installed", False):
                return {
                    "success": False,
                    "error": "xdotool not installed",
                    "output": f"xdotool is required for key input. {check_result.get('instructions', 'Please install xdotool manually.')}",
                    "instructions": check_result.get(
                        "instructions", "Please install xdotool manually."
                    ),
                }

            cmd = ["xdotool", "key", "--clearmodifiers", key]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": f"Pressed key: {key}",
                    "message": f"Successfully pressed {key}",
                }
            else:
                return {
                    "success": False,
                    "error": f"xdotool key failed: {result.stderr}",
                    "output": f"Failed to press {key}",
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Key press timed out"}
        except Exception as e:
            return {"success": False, "error": f"Error pressing key: {e}"}

    def execute_gui_actions(self, actions: list) -> Dict[str, Any]:
        """Execute a list of GUI actions (click, type, key_press)."""
        try:
            executed_actions = []
            failed_actions = []
            
            for action in actions:
                action_type = action.get("action", "").lower()
                coordinates = action.get("coordinates", {})
                x = coordinates.get("x", 0)
                y = coordinates.get("y", 0)
                
                try:
                    if action_type == "click":
                        result = self.click_screen(x=x, y=y, button="left")
                        if result.get("success"):
                            executed_actions.append(f"✅ Clicked at ({x}, {y})")
                        else:
                            failed_actions.append(f"❌ Failed to click at ({x}, {y}): {result.get('error', 'Unknown error')}")
                            
                    elif action_type == "type":
                        text = action.get("parameters", {}).get("text", "")
                        result = self.type_text(text=text, x=x, y=y)
                        if result.get("success"):
                            executed_actions.append(f"✅ Typed '{text}' at ({x}, {y})")
                        else:
                            failed_actions.append(f"❌ Failed to type '{text}' at ({x}, {y}): {result.get('error', 'Unknown error')}")
                            
                    elif action_type == "key_press":
                        key = action.get("parameters", {}).get("key", "enter")
                        result = self.press_key(key=key)
                        if result.get("success"):
                            executed_actions.append(f"✅ Pressed key '{key}'")
                        else:
                            failed_actions.append(f"❌ Failed to press key '{key}': {result.get('error', 'Unknown error')}")
                            
                except Exception as e:
                    failed_actions.append(f"❌ Error executing {action_type}: {str(e)}")

            execution_summary = "\n".join(executed_actions + failed_actions)
            
            return {
                "success": len(failed_actions) == 0,
                "executed_actions": executed_actions,
                "failed_actions": failed_actions,
                "total_executed": len(executed_actions),
                "total_failed": len(failed_actions),
                "output": f"Executed {len(executed_actions)} actions successfully, {len(failed_actions)} failed\n\n{execution_summary}",
                "message": f"GUI actions execution completed: {len(executed_actions)} successful, {len(failed_actions)} failed",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error executing GUI actions: {e}",
                "output": f"GUI actions execution failed: {e}",
            }

    def click_screen(self, x: int, y: int, button: str = "left") -> Dict[str, Any]:
        """Click at specific screen coordinates using xdotool with validation."""
        try:
            import subprocess

            # Enhanced coordinate validation
            if x < 0 or y < 0:
                return {
                    "success": False,
                    "error": f"Invalid coordinates: ({x}, {y}) - coordinates cannot be negative",
                    "output": f"Coordinates ({x}, {y}) cannot be negative",
                }

            # Check screen resolution to validate coordinates
            try:
                screen_info = subprocess.run(
                    ["xrandr"], capture_output=True, text=True, timeout=3
                )
                if screen_info.returncode == 0:
                    # Extract screen resolution from xrandr output
                    lines = screen_info.stdout.split("\n")
                    for line in lines:
                        if "*" in line and "connected" in line:
                            # Parse resolution like "1920x1080"
                            import re

                            resolution = re.search(r"(\d+)x(\d+)", line)
                            if resolution:
                                max_x, max_y = int(resolution.group(1)), int(
                                    resolution.group(2)
                                )
                                if x > max_x or y > max_y:
                                    return {
                                        "success": False,
                                        "error": f"Coordinates ({x}, {y}) exceed screen resolution ({max_x}x{max_y})",
                                        "output": f"Coordinates out of screen bounds",
                                    }
                                break
            except Exception as e:
                self.logger.warning(f"Could not validate screen resolution: {e}")

            # Check if xdotool is available
            check_result = self.check_system_dependency("xdotool")
            if not check_result.get("installed", False):
                return {
                    "success": False,
                    "error": "xdotool not installed",
                    "output": f"xdotool is required for screen clicking. {check_result.get('instructions', 'Please install xdotool manually.')}",
                    "instructions": check_result.get(
                        "instructions", "Please install xdotool manually."
                    ),
                }

            # Intelligently bring the target window to front before clicking
            try:
                # Get the current active window
                active_window_cmd = ["xdotool", "getactivewindow"]
                active_result = subprocess.run(
                    active_window_cmd, capture_output=True, text=True, timeout=2
                )

                if active_result.returncode == 0:
                    # Try to reactivate the current window to ensure it stays focused
                    window_activate_cmd = [
                        "xdotool",
                        "windowactivate",
                        active_result.stdout.strip(),
                    ]
                    subprocess.run(
                        window_activate_cmd, capture_output=True, text=True, timeout=2
                    )
            except:
                pass  # Continue even if window activation fails

            # Use xdotool to click directly at coordinates
            if button == "left":
                click_cmd = ["xdotool", "mousemove", str(x), str(y), "click", "1"]
            elif button == "right":
                click_cmd = ["xdotool", "mousemove", str(x), str(y), "click", "3"]
            elif button == "middle":
                click_cmd = ["xdotool", "mousemove", str(x), str(y), "click", "2"]
            else:
                click_cmd = [
                    "xdotool",
                    "mousemove",
                    str(x),
                    str(y),
                    "click",
                    "--button",
                    button,
                    "1",
                ]

            click_result = subprocess.run(
                click_cmd, capture_output=True, text=True, timeout=5
            )

            if click_result.returncode == 0:
                return {
                    "success": True,
                    "output": f"Clicked at coordinates ({x}, {y}) with {button} button",
                    "message": f"Successfully clicked at ({x}, {y})",
                }
            else:
                # Try alternative method with separate commands
                try:
                    move_cmd = ["xdotool", "mousemove", str(x), str(y)]
                    move_result = subprocess.run(
                        move_cmd, capture_output=True, text=True, timeout=3
                    )

                    if move_result.returncode == 0:
                        click_cmd = (
                            ["xdotool", "click", "1"]
                            if button == "left"
                            else ["xdotool", "click", "3"]
                        )
                        click_result = subprocess.run(
                            click_cmd, capture_output=True, text=True, timeout=3
                        )

                        if click_result.returncode == 0:
                            return {
                                "success": True,
                                "output": f"Clicked at coordinates ({x}, {y}) with {button} button (using move+click method)",
                                "message": f"Successfully clicked at ({x}, {y})",
                            }

                    return {
                        "success": False,
                        "error": f"xdotool click failed: {click_result.stderr}",
                        "output": f"Failed to click at ({x}, {y}) - {click_result.stderr}",
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Click operation failed: {e}",
                        "output": f"Failed to click at ({x}, {y})",
                    }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Click operation timed out"}
        except Exception as e:
            return {"success": False, "error": f"Error clicking screen: {e}"}

    def get_active_window(self) -> Dict[str, Any]:
        """Get information about the currently active/focused window."""
        try:
            import subprocess

            # Get active window ID
            active_cmd = ["xdotool", "getactivewindow"]
            active_result = subprocess.run(
                active_cmd, capture_output=True, text=True, timeout=3
            )

            if active_result.returncode != 0:
                return {
                    "success": False,
                    "error": "Could not get active window ID",
                    "output": "Failed to determine active window",
                }

            window_id = active_result.stdout.strip()

            # Get window name
            name_cmd = ["xdotool", "getwindowname", window_id]
            name_result = subprocess.run(
                name_cmd, capture_output=True, text=True, timeout=3
            )
            window_name = (
                name_result.stdout.strip() if name_result.returncode == 0 else "Unknown"
            )

            # Get window class
            class_cmd = ["xdotool", "getwindowclassname", window_id]
            class_result = subprocess.run(
                class_cmd, capture_output=True, text=True, timeout=3
            )
            window_class = (
                class_result.stdout.strip()
                if class_result.returncode == 0
                else "Unknown"
            )

            # Get window geometry
            geometry_cmd = ["xdotool", "getwindowgeometry", window_id]
            geometry_result = subprocess.run(
                geometry_cmd, capture_output=True, text=True, timeout=3
            )
            geometry_info = (
                geometry_result.stdout.strip()
                if geometry_result.returncode == 0
                else "Unknown"
            )

            detailed_output = f"Active window details:\n"
            detailed_output += f"  ID: {window_id}\n"
            detailed_output += f"  Name: '{window_name}'\n"
            detailed_output += f"  Class: '{window_class}'\n"
            detailed_output += f"  Geometry: {geometry_info}"
            
            return {
                "success": True,
                "window_id": window_id,
                "window_name": window_name,
                "window_class": window_class,
                "geometry": geometry_info,
                "output": detailed_output,
                "message": "Active window information retrieved successfully",
            }

        except Exception as e:
            return {"success": False, "error": f"Error getting active window: {e}"}

    def get_all_windows(self) -> Dict[str, Any]:
        """Get list of all visible windows with their details."""
        try:
            import subprocess

            # Get all window IDs
            search_cmd = ["xdotool", "search", "--onlyvisible", "--name", "."]
            search_result = subprocess.run(
                search_cmd, capture_output=True, text=True, timeout=5
            )

            if search_result.returncode != 0:
                return {
                    "success": False,
                    "error": "Could not get window list",
                    "output": "Failed to retrieve windows",
                }

            window_ids = search_result.stdout.strip().split("\n")
            windows = []

            for window_id in window_ids:
                if not window_id.strip():
                    continue

                try:
                    # Get window name
                    name_cmd = ["xdotool", "getwindowname", window_id]
                    name_result = subprocess.run(
                        name_cmd, capture_output=True, text=True, timeout=2
                    )
                    window_name = (
                        name_result.stdout.strip()
                        if name_result.returncode == 0
                        else "Unknown"
                    )

                    # Get window class
                    class_cmd = ["xdotool", "getwindowclassname", window_id]
                    class_result = subprocess.run(
                        class_cmd, capture_output=True, text=True, timeout=2
                    )
                    window_class = (
                        class_result.stdout.strip()
                        if class_result.returncode == 0
                        else "Unknown"
                    )

                    windows.append(
                        {"id": window_id, "name": window_name, "class": window_class}
                    )
                except:
                    continue

            # Format detailed window information
            window_details = []
            for i, window in enumerate(windows, 1):
                window_details.append(
                    f"{i}. ID: {window['id']} | Name: '{window['name']}' | Class: '{window['class']}'"
                )
            
            detailed_output = f"Found {len(windows)} visible windows:\n" + "\n".join(window_details)
            
            return {
                "success": True,
                "windows": windows,
                "count": len(windows),
                "output": detailed_output,
                "message": "Window list retrieved successfully",
            }

        except Exception as e:
            return {"success": False, "error": f"Error getting window list: {e}"}

    def focus_window(self, window_identifier: str = None, window_title: str = None, window_id: str = None, window_class: str = None) -> Dict[str, Any]:
        """
        Focus/activate a specific window by ID, name, or class.
        
        Args:
            window_identifier: General identifier (will be used if other params not provided)
            window_title: Specific window title to search for
            window_id: Specific window ID
            window_class: Specific window class
            
        Returns:
            Dict with success status and output
        """
        try:
            import subprocess
            
            # Determine which identifier to use
            identifier = window_id or window_title or window_class or window_identifier
            
            if not identifier:
                return {"success": False, "error": "No window identifier provided"}

            # Try different methods to find and focus the window
            methods = [
                # Method 1: Direct window ID
                lambda: subprocess.run(
                    ["xdotool", "windowactivate", identifier],
                    capture_output=True,
                    text=True,
                    timeout=3,
                ),
                # Method 2: Search by name and activate
                lambda: subprocess.run(
                    [
                        "xdotool",
                        "search",
                        "--name",
                        identifier,
                        "windowactivate",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                ),
                # Method 3: Search by class and activate
                lambda: subprocess.run(
                    [
                        "xdotool",
                        "search",
                        "--class",
                        identifier,
                        "windowactivate",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                ),
                # Method 4: Search by partial name match
                lambda: subprocess.run(
                    [
                        "xdotool",
                        "search",
                        "--onlyvisible",
                        "--name",
                        identifier,
                        "windowactivate",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                ),
            ]

            for i, method in enumerate(methods, 1):
                try:
                    result = method()
                    if result.returncode == 0:
                        return {
                            "success": True,
                            "output": f"Window focused using method {i} with identifier '{window_identifier}'",
                            "message": "Window focused successfully",
                        }
                except:
                    continue

            return {
                "success": False,
                "error": f"Could not focus window with identifier '{window_identifier}'",
                "output": "Tried multiple methods but window could not be found or focused",
            }

        except Exception as e:
            return {"success": False, "error": f"Error focusing window: {e}"}

    def bring_window_to_front(self, window_identifier: str = None) -> Dict[str, Any]:
        """Dynamically bring any window to the foreground based on intelligent detection."""
        try:
            import subprocess

            # If no specific identifier provided, try to detect the most relevant window
            if not window_identifier:
                # Get all visible windows and their titles
                try:
                    windows_cmd = ["wmctrl", "-l"]
                    windows_result = subprocess.run(
                        windows_cmd, capture_output=True, text=True, timeout=5
                    )

                    if windows_result.returncode == 0:
                        window_lines = windows_result.stdout.strip().split("\n")
                        # Try to activate windows in order of likelihood (most recent first)
                        for line in window_lines:
                            if line.strip() and len(line.split()) > 2:
                                window_id = line.split()[0]
                                try:
                                    activate_cmd = ["wmctrl", "-i", "-a", window_id]
                                    activate_result = subprocess.run(
                                        activate_cmd,
                                        capture_output=True,
                                        text=True,
                                        timeout=2,
                                    )
                                    if activate_result.returncode == 0:
                                        return {
                                            "success": True,
                                            "output": f"Window activated: {line}",
                                            "message": "Window brought to front",
                                        }
                                except:
                                    continue
                except:
                    pass

                # Fallback: try to activate the most recently used window
                try:
                    recent_cmd = ["xdotool", "getactivewindow", "windowactivate"]
                    recent_result = subprocess.run(
                        recent_cmd, capture_output=True, text=True, timeout=3
                    )
                    if recent_result.returncode == 0:
                        return {
                            "success": True,
                            "output": "Current active window reactivated",
                            "message": "Window focus maintained",
                        }
                except:
                    pass

                return {
                    "success": False,
                    "error": "No suitable window found to activate",
                    "output": "Could not identify a window to bring to front",
                }

            # If specific identifier provided, use the new focus_window method
            return self.focus_window(window_identifier)

        except Exception as e:
            return {"success": False, "error": f"Error bringing window to front: {e}"}

    def check_browser_status(self) -> Dict[str, Any]:
        """Check if browser is running and active."""
        try:
            import subprocess

            # Get all running processes
            ps_cmd = ["ps", "aux"]
            ps_result = subprocess.run(
                ps_cmd, capture_output=True, text=True, timeout=5
            )

            if ps_result.returncode != 0:
                return {"success": False, "error": "Could not get process list"}

            processes = ps_result.stdout.lower()

            # Look for browser-like applications dynamically
            browser_patterns = [
                "browser",
                "web",
                "mozilla",
                "webkit",
                "chromium",
                "safari",
                "edge",
                "opera",
            ]

            running_browsers = []
            for pattern in browser_patterns:
                if pattern in processes:
                    running_browsers.append(pattern)

            # Get active window title
            active_window_cmd = ["xdotool", "getactivewindow", "getwindowname"]
            active_window_result = subprocess.run(
                active_window_cmd, capture_output=True, text=True, timeout=3
            )
            active_window = (
                active_window_result.stdout.strip()
                if active_window_result.returncode == 0
                else ""
            )

            # Check if active window contains browser-like terms
            browser_active = any(
                browser in active_window.lower() for browser in browser_patterns
            )

            return {
                "success": True,
                "output": {
                    "browsers_running": running_browsers,
                    "browser_active": browser_active,
                    "active_window": active_window,
                    "any_browser_running": len(running_browsers) > 0,
                },
                "message": f"Browsers found: {', '.join(running_browsers) if running_browsers else 'None'}, Active: {active_window}",
            }

        except Exception as e:
            return {"success": False, "error": f"Error checking browser status: {e}"}

    def get_mouse_position(self) -> Dict[str, Any]:
        """Get current mouse position using xdotool."""
        try:
            import subprocess

            result = subprocess.run(
                ["xdotool", "getmouselocation", "--shell"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                # Parse the output
                lines = result.stdout.strip().split("\n")
                coords = {}
                for line in lines:
                    if "=" in line:
                        key, value = line.split("=", 1)
                        coords[key] = int(value)

                return {
                    "success": True,
                    "output": f"Mouse position: X={coords.get('X', 0)}, Y={coords.get('Y', 0)}",
                    "x": coords.get("X", 0),
                    "y": coords.get("Y", 0),
                    "message": f"Mouse is at ({coords.get('X', 0)}, {coords.get('Y', 0)})",
                }
            else:
                return {
                    "success": False,
                    "error": f"xdotool failed: {result.stderr}",
                    "output": "Failed to get mouse position",
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Mouse position query timed out"}
        except Exception as e:
            return {"success": False, "error": f"Error getting mouse position: {e}"}

    def read_screen(
        self,
        prompt: str = "Describe what you see on this screen in detail",
        force_new: bool = False,
    ) -> Dict[str, Any]:
        """Capture the current screen, analyze it with AI, and clean up the temporary image with intelligent caching."""
        try:
            import tempfile
            import os
            import time
            from src.core.gemini_client import GeminiClient

            # Check if we should use cached analysis (unless forced)
            if not force_new and hasattr(self, "_last_screenshot_time"):
                time_since_last = time.time() - self._last_screenshot_time
                if time_since_last < 3:  # Less than 3 seconds since last screenshot
                    return {
                        "success": True,
                        "output": "Using recent screenshot analysis (cached)",
                        "text": getattr(self, "_last_screenshot_text", ""),
                        "message": "Recent analysis reused to avoid redundancy",
                    }

            # Create temporary file for screenshot
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                screenshot_path = temp_file.name

            # Take screenshot using system command
            if os.name == "nt":  # Windows
                # Use Windows Snipping Tool or PowerShell
                screenshot_cmd = f"powershell -Command \"Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Screen]::PrimaryScreen.Bounds | ForEach-Object {{ $bounds = $_; $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height; $graphics = [System.Drawing.Graphics]::FromImage($bitmap); $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size); $bitmap.Save('{screenshot_path}', [System.Drawing.Imaging.ImageFormat]::Png); $bitmap.Dispose(); $graphics.Dispose() }}\""
            else:  # Linux/macOS
                # Use scrot (Linux) or screencapture (macOS)
                if os.path.exists("/usr/bin/scrot"):
                    screenshot_cmd = f'scrot "{screenshot_path}"'
                elif os.path.exists("/usr/bin/gnome-screenshot"):
                    screenshot_cmd = f'gnome-screenshot -f "{screenshot_path}"'
                elif os.path.exists("/usr/bin/import"):
                    screenshot_cmd = f'import -window root "{screenshot_path}"'
                else:
                    # Try screencapture for macOS
                    screenshot_cmd = f'screencapture "{screenshot_path}"'

            # Execute screenshot command
            screenshot_result = self.run_shell(screenshot_cmd)

            if not screenshot_result.get("success", False):
                return {
                    "success": False,
                    "error": f"Failed to capture screenshot: {screenshot_result.get('error', 'Unknown error')}",
                    "output": "Screenshot capture failed",
                }

            # Check if screenshot file was created and has content
            if (
                not os.path.exists(screenshot_path)
                or os.path.getsize(screenshot_path) == 0
            ):
                return {
                    "success": False,
                    "error": "Screenshot file was not created or is empty",
                    "output": "Screenshot capture failed - no image data",
                }

            # Initialize Gemini client and analyze the screenshot
            gemini_client = GeminiClient()
            analysis_result = gemini_client.analyze_image(screenshot_path, prompt)

            # Clean up temporary screenshot file
            try:
                os.unlink(screenshot_path)
            except Exception as cleanup_error:
                self.logger.warning(
                    f"Failed to clean up temporary screenshot: {cleanup_error}"
                )

            if analysis_result:
                # Extract the text content from the analysis result
                if isinstance(analysis_result, dict):
                    analysis_text = analysis_result.get("text", str(analysis_result))
                else:
                    analysis_text = str(analysis_result)

                # Cache the analysis for future use
                self._last_screenshot_time = time.time()
                self._last_screenshot_text = analysis_text

                # Format the analysis result for better readability
                formatted_output = self._format_screen_analysis(analysis_text)

                return {
                    "success": True,
                    "text": analysis_text,
                    "output": formatted_output,
                    "message": "Screen captured and analyzed successfully",
                    "summary": f"Screen analysis completed: {len(analysis_text)} characters analyzed",
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to analyze screenshot",
                    "output": "No analysis result returned",
                }

        except Exception as e:
            # Clean up temporary file if it exists
            try:
                if "screenshot_path" in locals() and os.path.exists(screenshot_path):
                    os.unlink(screenshot_path)
            except:
                pass

            return {
                "success": False,
                "error": f"Error reading screen: {e}",
                "output": f"Error: {e}",
            }

    def install_system_package(self, package_name: str) -> Dict[str, Any]:
        """Install system packages using the appropriate package manager."""
        try:
            import subprocess
            import platform
            import os

            system = platform.system().lower()

            if system == "linux":
                if os.path.exists("/usr/bin/apt"):
                    cmd = f"sudo apt update && sudo apt install -y {package_name}"
                    manager = "apt"
                elif os.path.exists("/usr/bin/yum"):
                    cmd = f"sudo yum install -y {package_name}"
                    manager = "yum"
                elif os.path.exists("/usr/bin/dnf"):
                    cmd = f"sudo dnf install -y {package_name}"
                    manager = "dnf"
                elif os.path.exists("/usr/bin/pacman"):
                    cmd = f"sudo pacman -S --noconfirm {package_name}"
                    manager = "pacman"
                elif os.path.exists("/usr/bin/zypper"):
                    cmd = f"sudo zypper install -y {package_name}"
                    manager = "zypper"
                else:
                    return {
                        "success": False,
                        "error": "No supported package manager found",
                        "output": f"Please install {package_name} manually. Supported managers: apt, yum, dnf, pacman, zypper",
                    }
            elif system == "darwin":  # macOS
                if os.path.exists("/usr/local/bin/brew"):
                    cmd = f"brew install {package_name}"
                    manager = "homebrew"
                else:
                    return {
                        "success": False,
                        "error": "Homebrew not found",
                        "output": f'Please install Homebrew first: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" then run: brew install {package_name}',
                    }
            elif system == "windows":
                if os.path.exists("C:\\ProgramData\\chocolatey\\bin\\choco.exe"):
                    cmd = f"choco install {package_name} -y"
                    manager = "chocolatey"
                else:
                    return {
                        "success": False,
                        "error": "Chocolatey not found",
                        "output": f"Please install Chocolatey first: https://chocolatey.org/install then run: choco install {package_name}",
                    }
            else:
                return {
                    "success": False,
                    "error": f"Unsupported operating system: {system}",
                    "output": f"Please install {package_name} manually for {system}",
                }

            result = self.run_shell(cmd)

            if result["success"]:
                return {
                    "success": True,
                    "output": f"Package '{package_name}' installed successfully using {manager}",
                    "message": f"Successfully installed {package_name} using {manager}",
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to install {package_name} using {manager}",
                    "output": f"Installation failed: {result.get('error', 'Unknown error')}. Manual installation required.",
                    "manual_command": cmd,
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error installing system package: {e}",
                "output": f"Please install {package_name} manually",
            }

    def check_system_dependency(
        self, dependency: str = None, dependency_name: str = None
    ) -> Dict[str, Any]:
        """Check if a system dependency is installed and provide installation instructions if not."""
        try:
            import subprocess
            import platform
            import os

            # Handle both parameter names
            dep_name = dependency or dependency_name
            if not dep_name:
                return {
                    "success": False,
                    "error": "No dependency name provided",
                    "output": "Please provide either 'dependency' or 'dependency_name' parameter",
                }

            result = subprocess.run(["which", dep_name], capture_output=True, text=True)

            if result.returncode == 0:
                version_result = subprocess.run(
                    [dep_name, "--version"], capture_output=True, text=True
                )
                version = (
                    version_result.stdout.strip()
                    if version_result.returncode == 0
                    else "Unknown"
                )

                return {
                    "success": True,
                    "installed": True,
                    "path": result.stdout.strip(),
                    "version": version,
                    "output": f"{dep_name} is installed at {result.stdout.strip()}",
                }
            else:
                system = platform.system().lower()
                instructions = self._get_installation_instructions(dep_name, system)

                return {
                    "success": False,
                    "installed": False,
                    "error": f"{dep_name} is not installed",
                    "output": f"{dep_name} not found. Installation instructions: {instructions}",
                    "instructions": instructions,
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error checking dependency: {e}",
                "output": f"Could not check if {dep_name} is installed",
            }

    def _get_installation_instructions(self, dependency: str, system: str) -> str:
        """Get installation instructions for a dependency based on the operating system."""
        instructions = {
            "linux": {
                "xdotool": "sudo apt install xdotool (Ubuntu/Debian) or sudo yum install xdotool (RHEL/CentOS)",
                "scrot": "sudo apt install scrot (Ubuntu/Debian) or sudo yum install scrot (RHEL/CentOS)",
                "gnome-screenshot": "sudo apt install gnome-screenshot (Ubuntu/Debian) or sudo yum install gnome-screenshot (RHEL/CentOS)",
                "import": "sudo apt install imagemagick (Ubuntu/Debian) or sudo yum install ImageMagick (RHEL/CentOS)",
                "pylint": "pip install pylint (in virtual environment)",
                "pyautogui": "pip install pyautogui (in virtual environment)",
            },
            "darwin": {
                "screencapture": "Built-in on macOS",
                "xdotool": "brew install xdotool",
                "pylint": "pip install pylint (in virtual environment)",
                "pyautogui": "pip install pyautogui (in virtual environment)",
            },
            "windows": {
                "xdotool": "Not available on Windows. Use pyautogui instead.",
                "pylint": "pip install pylint (in virtual environment)",
                "pyautogui": "pip install pyautogui (in virtual environment)",
            },
        }

        return instructions.get(system, {}).get(
            dependency, f"Please install {dependency} manually for {system}"
        )

    def analyze_screen_actions(
        self, task_description: str = "Analyze the screen and provide actionable steps"
    ) -> Dict[str, Any]:
        """Analyze the screen and provide actionable steps with coordinates and operations."""
        try:
            import tempfile
            import os
            from src.core.gemini_client import GeminiClient

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                screenshot_path = temp_file.name

            if os.name == "nt":  # Windows
                screenshot_cmd = f"powershell -Command \"Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Screen]::PrimaryScreen.Bounds | ForEach-Object {{ $bounds = $_; $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height; $graphics = [System.Drawing.Graphics]::FromImage($bitmap); $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size); $bitmap.Save('{screenshot_path}', [System.Drawing.Imaging.ImageFormat]::Png); $bitmap.Dispose(); $graphics.Dispose() }}\""
            else:  # Linux/macOS
                if os.path.exists("/usr/bin/scrot"):
                    screenshot_cmd = f'scrot "{screenshot_path}"'
                elif os.path.exists("/usr/bin/gnome-screenshot"):
                    screenshot_cmd = f'gnome-screenshot -f "{screenshot_path}"'
                elif os.path.exists("/usr/bin/import"):
                    screenshot_cmd = f'import -window root "{screenshot_path}"'
                else:
                    screenshot_cmd = f'screencapture "{screenshot_path}"'

            screenshot_result = self.run_shell(screenshot_cmd)

            if not screenshot_result.get("success", False):
                return {
                    "success": False,
                    "error": f"Failed to capture screenshot: {screenshot_result.get('error', 'Unknown error')}",
                    "output": "Screenshot capture failed",
                }

            if (
                not os.path.exists(screenshot_path)
                or os.path.getsize(screenshot_path) == 0
            ):
                return {
                    "success": False,
                    "error": "Screenshot file was not created or is empty",
                    "output": "Screenshot capture failed - no image data",
                }

            # Create detailed prompt for action analysis with strict validation
            action_prompt = f"""
Analyze this screenshot and provide actionable steps to complete this task: "{task_description}"

CRITICAL VALIDATION RULES:
1. ONLY analyze what you can actually SEE on the screen
2. DO NOT assume actions have been completed unless you can clearly see the result
3. DO NOT make up coordinates or actions that aren't clearly visible
4. If you cannot clearly see a browser window, DO NOT suggest browser-related actions
5. If the screen is black, blank, or shows desktop, DO NOT suggest clicking on non-existent elements

VIDEO PLATFORM RULES:
- On video search results: Click on VIDEO THUMBNAILS to play videos (no play button needed)
- Video thumbnails are the clickable areas that start video playback
- Look for video titles, thumbnails, and channel names as click targets
- Don't look for play buttons on search results - they don't exist

STRICT ANALYSIS REQUIREMENTS:
- If no browser is visible: suggest check_browser_status first
- If browser is visible but task not completed: suggest specific visible actions
- If task appears completed: confirm what you actually see completed
- NEVER fabricate actions or coordinates
- For video platforms: focus on clicking video thumbnails, not play buttons

Please provide a JSON response with the following structure:
{{
    "analysis": "EXACT description of what you can actually see on the screen",
    "browser_status": {{
        "browser_visible": true/false (only true if you can clearly see a browser window),
        "browser_active": true/false (only true if browser is clearly in foreground),
        "page_loaded": true/false (only true if you can see a loaded webpage),
        "recommendations": "Specific recommendations based on what you actually see"
    }},
    "actions": [
        {{
            "step": 1,
            "action": "check_browser_status|focus_window|click|right_click|double_click|scroll_up|scroll_down|scroll_left|scroll_right|drag|type|key_press",
            "coordinates": [x, y] (ONLY if you can clearly see the target element),
            "description": "What this action will do based on what you can see",
            "parameters": {{
                "text": "text to type (for type action)",
                "key": "key to press (for key_press action)",
                "duration": "drag duration in seconds (for drag action)"
            }}
        }}
    ],
    "confidence": 0.85,
    "notes": "Warnings about what you cannot see or verify on screen"
}}

AVAILABLE ACTIONS (use only what makes sense based on what you can see):
- check_browser_status: Check if browser is running and active (use first if no browser visible)
- focus_window: Focus specific window (use if browser visible but not active)
- click: Left mouse click at coordinates (ONLY if you can see the target element)
- right_click: Right mouse click at coordinates (ONLY if you can see the target element)
- double_click: Double click at coordinates (ONLY if you can see the target element)
- scroll_up: Scroll up from coordinates
- scroll_down: Scroll down from coordinates
- scroll_left: Scroll left from coordinates
- scroll_right: Scroll right from coordinates
- drag: Drag from coordinates to another location
- type: Type text at coordinates (ONLY if you can see a text input field)
- key_press: Press a key combination

REMEMBER: Be extremely conservative. Only suggest actions for elements you can clearly see and identify on the screen.
"""

            gemini_client = GeminiClient()
            analysis_result = gemini_client.analyze_image(
                screenshot_path, action_prompt
            )

            try:
                os.unlink(screenshot_path)
            except Exception as cleanup_error:
                self.logger.warning(
                    f"Failed to clean up temporary screenshot: {cleanup_error}"
                )

            if analysis_result:
                # Extract the text content from the analysis result
                if isinstance(analysis_result, dict):
                    analysis_text = analysis_result.get("text", str(analysis_result))
                else:
                    analysis_text = str(analysis_result)

                # Try to parse JSON from the response
                try:
                    import json
                    import re

                    # Extract JSON from the response
                    json_match = re.search(r"\{.*\}", analysis_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        actions_data = json.loads(json_str)

                        # Validate the analysis before returning
                        validated_actions = self._validate_analysis(
                            actions_data, task_description
                        )

                        formatted_output = self._format_action_analysis(
                            validated_actions
                        )

                        # Execute the actions if they are simple and safe
                        actions_list = validated_actions.get("actions", []) if isinstance(validated_actions, dict) else []
                        executed_actions = []
                        
                        for action in actions_list:
                            action_type = action.get("action", "").lower()
                            coordinates = action.get("coordinates", {})
                            x = coordinates.get("x", 0)
                            y = coordinates.get("y", 0)
                            
                            try:
                                if action_type == "click":
                                    # Execute click
                                    click_result = self.click_screen(x=x, y=y, button="left")
                                    if click_result.get("success"):
                                        executed_actions.append(f"✅ Clicked at ({x}, {y})")
                                    else:
                                        executed_actions.append(f"❌ Failed to click at ({x}, {y})")
                                        
                                elif action_type == "type":
                                    # Execute typing
                                    text = action.get("parameters", {}).get("text", "")
                                    type_result = self.type_text(text=text, x=x, y=y)
                                    if type_result.get("success"):
                                        executed_actions.append(f"✅ Typed '{text}' at ({x}, {y})")
                                    else:
                                        executed_actions.append(f"❌ Failed to type '{text}' at ({x}, {y})")
                                        
                                elif action_type == "key_press":
                                    # Execute key press
                                    key = action.get("parameters", {}).get("key", "enter")
                                    key_result = self.press_key(key=key)
                                    if key_result.get("success"):
                                        executed_actions.append(f"✅ Pressed key '{key}'")
                                    else:
                                        executed_actions.append(f"❌ Failed to press key '{key}'")
                                        
                            except Exception as e:
                                executed_actions.append(f"❌ Error executing {action_type}: {str(e)}")

                        execution_summary = "\n".join(executed_actions) if executed_actions else "No actions executed"
                        enhanced_output = f"{formatted_output}\n\n🎬 EXECUTION RESULTS:\n{execution_summary}"

                        return {
                            "success": True,
                            "text": analysis_text,
                            "output": enhanced_output,
                            "message": "Screen analyzed and actions executed successfully",
                            "summary": f"Generated {len(actions_list)} validated actionable steps and executed them",
                            "actions": actions_list,
                            "executed_actions": executed_actions,
                            "execution_summary": execution_summary,
                            "raw_analysis": analysis_text,
                            "validation_notes": validated_actions.get(
                                "validation_notes", []
                            ) if isinstance(validated_actions, dict) else [],
                        }
                    else:
                        # If no JSON found, return formatted text
                        formatted_output = self._format_action_analysis_text(
                            analysis_text
                        )
                        return {
                            "success": True,
                            "text": analysis_text,
                            "output": formatted_output,
                            "message": "Screen analyzed but no structured actions found",
                            "summary": "Analysis completed - manual interpretation required",
                        }

                except json.JSONDecodeError as e:
                    # If JSON parsing fails, return formatted text
                    formatted_output = self._format_action_analysis_text(analysis_text)
                    return {
                        "success": True,
                        "text": analysis_text,
                        "output": formatted_output,
                        "message": "Screen analyzed but JSON parsing failed",
                        "summary": "Analysis completed - manual interpretation required",
                    }
            else:
                return {
                    "success": False,
                    "error": "Failed to analyze screenshot",
                    "output": "No analysis result returned",
                }

        except Exception as e:
            try:
                if "screenshot_path" in locals() and os.path.exists(screenshot_path):
                    os.unlink(screenshot_path)
            except:
                pass

            return {
                "success": False,
                "error": f"Error analyzing screen actions: {e}",
                "output": f"Error: {e}",
            }

    def _validate_analysis(
        self, analysis_data: Dict[str, Any], task_description: str
    ) -> Dict[str, Any]:
        """Validate the analysis data to prevent fake results."""
        validation_notes = []
        validated_actions = []

        # Check browser status
        browser_status = analysis_data.get("browser_status", {})
        if browser_status.get("browser_visible", False):
            if not browser_status.get("browser_active", False):
                validation_notes.append(
                    "Browser visible but not active - may need focus_window"
                )

        # Validate actions
        actions = analysis_data.get("actions", [])
        for action in actions:
            action_type = action.get("action", "")
            coordinates = action.get("coordinates", [])

            # Validate coordinates
            if (
                action_type in ["click", "right_click", "double_click", "type"]
                and coordinates
            ):
                if len(coordinates) != 2:
                    validation_notes.append(
                        f"Invalid coordinates for {action_type}: {coordinates}"
                    )
                    continue

                x, y = coordinates
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    validation_notes.append(
                        f"Non-numeric coordinates for {action_type}: {coordinates}"
                    )
                    continue

                if x < 0 or y < 0 or x > 4000 or y > 4000:  # Reasonable screen bounds
                    validation_notes.append(
                        f"Coordinates out of reasonable bounds for {action_type}: {coordinates}"
                    )
                    continue

            # Check for suspicious patterns (fake results)
            description = action.get("description", "").lower()
            if any(
                phrase in description
                for phrase in [
                    "opened",
                    "search bar clicked",
                    "typed",
                    "playing",
                ]
            ):
                if not browser_status.get("browser_visible", False):
                    validation_notes.append(
                        f"Suspicious action '{action_type}' without visible browser"
                    )
                    continue

            # Video platform validation - clicking thumbnails is normal
            if (
                any(
                    platform in task_description.lower()
                    for platform in ["video", "play"]
                )
                and action_type == "click"
            ):
                if any(
                    phrase in description
                    for phrase in ["thumbnail", "video", "click on video", "play video"]
                ):
                    # This is a valid video action, don't filter it out
                    pass

            validated_actions.append(action)

        # If no browser is visible but task involves web interaction, add browser check first
        if (
            any(
                term in task_description.lower()
                for term in ["video", "web", "browser", "search"]
            )
            and "browser" not in task_description.lower()
        ):
            if not browser_status.get("browser_visible", False):
                validated_actions.insert(
                    0,
                    {
                        "step": 1,
                        "action": "check_browser_status",
                        "coordinates": None,
                        "description": "Check if browser is running and active",
                        "parameters": {},
                    },
                )

        # Update the analysis data
        analysis_data["actions"] = validated_actions
        analysis_data["validation_notes"] = validation_notes

        return analysis_data

    def _format_action_analysis(self, actions_data: dict) -> str:
        """Format action analysis data for better readability."""
        try:
            analysis = actions_data.get("analysis", "No analysis provided")
            actions = actions_data.get("actions", [])
            confidence = actions_data.get("confidence", 0)
            notes = actions_data.get("notes", "")

            formatted_output = f"""🎯 SCREEN ACTION ANALYSIS
{'='*60}

📋 ANALYSIS: {analysis}

🎬 ACTIONABLE STEPS:
{'='*60}"""

            for action in actions:
                step = action.get("step", "?")
                action_type = action.get("action", "unknown")
                coords = action.get("coordinates", [0, 0])
                description = action.get("description", "No description")
                params = action.get("parameters", {})

                formatted_output += f"""
📌 STEP {step}: {action_type.upper()}
   Coordinates: ({coords[0]}, {coords[1]})
   Description: {description}"""

                if params:
                    formatted_output += "\n   Parameters:"
                    for key, value in params.items():
                        formatted_output += f"\n     • {key}: {value}"

            formatted_output += f"""

📊 CONFIDENCE: {confidence * 100:.0f}%

💡 NOTES: {notes}

{'='*60}
✅ Action analysis completed successfully"""

            return formatted_output

        except Exception as e:
            return f"Error formatting action analysis: {e}"

    def _format_action_analysis_text(self, analysis_text: str) -> str:
        """Format action analysis text when JSON parsing fails."""
        return f"""🎯 SCREEN ACTION ANALYSIS
{'='*60}

{analysis_text}

{'='*60}
✅ Analysis completed - manual interpretation required"""

    def _format_screen_analysis(self, analysis_text: str) -> str:
        """Format screen analysis text for better readability."""
        try:
            # Split the analysis into sections
            lines = analysis_text.split("\n")
            formatted_lines = []

            for line in lines:
                original_line = line
                line = line.strip()
                if not line:
                    formatted_lines.append("")
                    continue

                # Format headers and subheaders
                if (
                    line.startswith("**")
                    and line.endswith("**")
                    and "**" not in line[2:-2]
                ):
                    # Main header (bold text)
                    header_text = line[2:-2]
                    formatted_lines.append(f"\n{'='*60}")
                    formatted_lines.append(f"📋 {header_text.upper()}")
                    formatted_lines.append(f"{'='*60}")
                elif line.startswith("*   **") and line.endswith("**"):
                    # Subheader with bullet
                    subheader_text = line[6:-2]
                    formatted_lines.append(f"\n🔹 {subheader_text}")
                    formatted_lines.append("-" * 40)
                elif line.startswith("*   "):
                    # Regular bullet point
                    bullet_text = line[4:]
                    formatted_lines.append(f"  • {bullet_text}")
                elif line.startswith("1.  **") and line.endswith("**"):
                    # Numbered item with bold
                    item_text = line[6:-2]
                    formatted_lines.append(f"\n📌 {item_text}")
                elif line.startswith("2.  **") and line.endswith("**"):
                    # Numbered item with bold
                    item_text = line[6:-2]
                    formatted_lines.append(f"\n📌 {item_text}")
                elif line.startswith("3.  **") and line.endswith("**"):
                    # Numbered item with bold
                    item_text = line[6:-2]
                    formatted_lines.append(f"\n📌 {item_text}")
                elif line.startswith("4.  **") and line.endswith("**"):
                    # Numbered item with bold
                    item_text = line[6:-2]
                    formatted_lines.append(f"\n📌 {item_text}")
                elif line.startswith("5.  **") and line.endswith("**"):
                    # Numbered item with bold
                    item_text = line[6:-2]
                    formatted_lines.append(f"\n📌 {item_text}")
                elif line.startswith("    *   "):
                    # Sub-bullet point
                    sub_bullet_text = line[8:]
                    formatted_lines.append(f"    ◦ {sub_bullet_text}")
                elif line.startswith("    *   **") and line.endswith("**"):
                    # Sub-bullet with bold
                    sub_bullet_text = line[10:-2]
                    formatted_lines.append(f"    ◦ {sub_bullet_text}")
                elif line.startswith("    *   "):
                    # Sub-bullet point (alternative format)
                    sub_bullet_text = line[8:]
                    formatted_lines.append(f"    ◦ {sub_bullet_text}")
                else:
                    # Regular text - preserve original indentation
                    if original_line.startswith("    "):
                        formatted_lines.append(f"    {line}")
                    elif original_line.startswith("  "):
                        formatted_lines.append(f"  {line}")
                    else:
                        formatted_lines.append(f"  {line}")

            # Join all lines and add some spacing
            formatted_text = "\n".join(formatted_lines)

            # Add a header and footer
            final_output = f"""🖥️  SCREEN ANALYSIS REPORT
{'='*60}

{formatted_text}

{'='*60}
✅ Analysis completed successfully"""

            return final_output

        except Exception as e:
            # If formatting fails, return the original text with a simple header
            return f"""🖥️  SCREEN ANALYSIS
{'='*50}

{analysis_text}

{'='*50}
✅ Analysis completed"""
