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
    
    def run_shell(self, command: str) -> Dict[str, Any]:
        """
        Execute a shell command with robust error handling and validation.
        
        Args:
            command: The shell command to execute
            
        Returns:
            Dict with success status and output/error
        """
        try:
            # Validate input
            if not command or not command.strip():
                return {"success": False, "error": "Empty or invalid command provided"}
            
            command = command.strip()
            self.logger.info(f"Executing shell command: `{command}`")
            
            # Enhanced security checks
            dangerous_patterns = [
                r'rm\s+-rf\s+/',  # Dangerous rm commands
                r'mkfs\.',        # Format commands
                r'dd\s+if=',      # Disk operations
                r'>\s*/dev/',     # Redirecting to device files
                r'chmod\s+777',   # Dangerous permissions
                r'chown\s+-R\s+root',  # Dangerous ownership changes
                r'sudo\s+rm',     # Sudo with rm
                r'su\s+-',        # Switch user
                r'passwd',        # Password changes
                r'shutdown',      # System shutdown
                r'reboot',        # System reboot
                r'halt',          # System halt
                r'poweroff',      # Power off
                r'init\s+0',      # Init level 0
                r'killall',       # Kill all processes
                r'pkill\s+-9',    # Force kill
                r'kill\s+-9\s+-1', # Kill all processes
            ]
            
            import re
            for pattern in dangerous_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return {"success": False, "error": f"Access denied: Potentially dangerous command detected: {command}"}
            
            # Check for system directory access
            dangerous_dirs = ['/root', '/etc', '/sys', '/proc', '/dev', '/var', '/usr', '/bin', '/sbin', '/lib', '/lib64']
            for dangerous_dir in dangerous_dirs:
                if f'cd {dangerous_dir}' in command or f'cd {dangerous_dir}/' in command:
                    return {"success": False, "error": f"Access denied: System directory access not allowed: {command}"}
            
            # Auto-quote file paths with spaces for Linux compatibility
            # Pattern to match file paths with spaces
            path_pattern = r'/[^\s]*\s[^\s]*'
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
                timeout=300,
                cwd=settings.BASE_DIR,
                env=os.environ.copy()  # Preserve environment
            )
            execution_time = time.time() - start_time
            
            # Capture output for return value with better error handling
            captured_result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=settings.BASE_DIR,
                env=os.environ.copy()
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
                "command": command
            }
            
            if captured_result.returncode == 0:
                self.logger.success(f"Command completed successfully in {execution_time:.2f}s")
            else:
                self.logger.error(f"Command failed with return code {captured_result.returncode}")
                result_dict["error"] = output
            
            return result_dict
            
        except subprocess.TimeoutExpired as e:
            error_msg = "Command timed out after 300 seconds"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "command": command}
        except subprocess.CalledProcessError as e:
            error_msg = f"Command failed with CalledProcessError: {e}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "return_code": e.returncode, "command": command}
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
    
    
    def install_package(self, package_name: str) -> Dict[str, Any]:
        """
        Install a Python package using pip in the current virtual environment.
        
        Args:
            package_name: The name of the package to install
            
        Returns:
            Dict with success status and output/error
        """
        import os
        
        venv_path = os.environ.get('VIRTUAL_ENV')
        if not venv_path:
            return {
                "success": False,
                "error": "No virtual environment detected. Please activate the venv first.",
                "output": "Virtual environment required for package installation"
            }
        
        self.logger.info(f"Installing package: {package_name} in venv")
        return self.run_shell(f"python3 -m pip install {package_name}")
    
    def read_file(self, file_path: str, encoding: str = "utf-8", max_size_mb: float = 100.0) -> Dict[str, Any]:
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
                return {"success": False, "error": "Empty or invalid file path provided"}
            
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
                return {"success": False, "error": f"File too large: {file_size / (1024*1024):.1f}MB (max {max_size_mb}MB)"}
            
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
                    self.logger.debug(f"Failed to read with encoding {enc}, trying next...")
                    continue
                except Exception as e:
                    self.logger.debug(f"Error reading with encoding {enc}: {e}")
                    continue
            
            if content is None:
                return {"success": False, "error": f"Failed to read file with any supported encoding: {file_path}"}
            
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
                "characters": len(content) if content else 0
            }
            
            self.logger.success(f"Successfully read file: {file_path} ({file_size / 1024:.1f}KB)")
            return result
            
        except PermissionError as e:
            return {"success": False, "error": f"Permission denied reading file: {e}"}
        except OSError as e:
            return {"success": False, "error": f"OS error reading file: {e}"}
        except Exception as e:
            self.logger.error(f"Unexpected error reading file: {e}")
            return {"success": False, "error": f"Failed to read file: {str(e)}"}
    
    def write_file(self, file_path: str, content: str, encoding: str = "utf-8", create_backup: bool = False) -> Dict[str, Any]:
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
                return {"success": False, "error": "Empty or invalid file path provided"}
            
            if content is None:
                content = ""
            
            file_path = file_path.strip()
            self.logger.debug(f"Writing file: {file_path}")
            
            # Resolve path
            path = self._resolve_path(file_path)
            
            # Check if file already exists and create backup if requested
            if path.exists() and create_backup:
                backup_path = path.with_suffix(path.suffix + '.backup')
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
                return {"success": False, "error": f"Permission denied creating directory: {e}"}
            except Exception as e:
                return {"success": False, "error": f"Failed to create directory: {e}"}
            
            # Check if we can write to the location
            if path.exists() and not os.access(path, os.W_OK):
                return {"success": False, "error": f"File not writable: {file_path}"}
            
            # Check if parent directory is writable
            if not os.access(path.parent, os.W_OK):
                return {"success": False, "error": f"Directory not writable: {path.parent}"}
            
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
                    return {"success": False, "error": f"Failed to write file with any encoding: {e2}"}
            except Exception as e:
                return {"success": False, "error": f"Failed to write file: {e}"}
            
            # Verify the file was written correctly
            if not path.exists():
                return {"success": False, "error": "File was not created after write operation"}
            
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
                "backup_created": create_backup and path.exists()
            }
            
            self.logger.success(f"Successfully wrote file: {file_path} ({file_size / 1024:.1f}KB)")
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
                return {"success": False, "error": f"Path is not a directory: {directory}"}
            
            # Check if directory is readable
            if not os.access(path, os.R_OK):
                return {"success": False, "error": f"Directory not readable: {directory}"}
            
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
                            "modified": item_stat.st_mtime
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
                        output_lines.append(item['path'])
                
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
                    "directories_list": [item["name"] for item in directories]
                }
                
                self.logger.success(f"Successfully listed directory: {directory} ({len(items)} items)")
                return result
                
            except PermissionError as e:
                return {"success": False, "error": f"Permission denied listing directory: {e}"}
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
            if not (path.is_relative_to(base_dir) or path.is_relative_to(Path.home()) or str(path) == '.' or str(path) == '..'):
                # Check if it's a relative path that should be allowed
                if not (str(directory).startswith('.') or str(directory).startswith('~')):
                    return {"success": False, "error": f"Access denied: Directory outside allowed scope: {directory}"}
            
            # Block access to critical system directories only
            blocked_dirs = ['/etc', '/sys', '/proc', '/dev', '/var', '/usr', '/root', '/bin', '/sbin', '/lib', '/lib64']
            if any(str(path).startswith(blocked_dir) for blocked_dir in blocked_dirs):
                return {"success": False, "error": f"Access denied: System directory access not allowed: {directory}"}
            
            # Block access to root directory only
            if str(path) == '/':
                return {"success": False, "error": f"Access denied: Root directory access not allowed: {directory}"}
            
            if not path.exists():
                return {"success": False, "error": f"Directory not found: {directory}"}
            
            if not path.is_dir():
                return {"success": False, "error": f"Path is not a directory: {directory}"}
            
            # Change directory
            os.chdir(path)
            
            # Update context manager with new directory
            if hasattr(self.system, 'context_manager'):
                self.system.context_manager.add_context_entry(
                    "directory_change", 
                    f"Changed directory to: {path}",
                    {"new_directory": str(path)}
                )
            
            return {"success": True, "output": f"Changed directory to: {path}", "current_dir": str(path)}
        except Exception as e:
            return {"success": False, "error": f"Failed to change directory: {e}"}
    
    def create_and_save_tool(self, tool_name: str, tool_code: str, doc_string: str) -> Dict[str, Any]:
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
                "datetime": datetime
            }
            exec(tool_code, globals(), exec_context)
            func = exec_context.get(tool_name)
            
            if not func:
                return {
                    "success": False, 
                    "error": f"Function '{tool_name}' not found in the provided code."
                }
            
            # Create tool object
            from database.memory import Tool
            new_tool = Tool(
                name=tool_name,
                code=tool_code,
                doc=doc_string,
                is_dynamic=True,
                last_used=time.time(),
                func=func
            )
            
            # Register with system if available
            if self.system and hasattr(self.system, 'tool_manager'):
                self.system.tool_manager.register_tool(new_tool)
            
            # Save to file
            tool_path = settings.TOOLS_DIR / f"{tool_name}.py"
            with open(tool_path, "w", encoding="utf-8") as f:
                f.write(tool_code)
            
            return {
                "success": True, 
                "output": f"Tool '{tool_name}' created and registered successfully."
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
                    "working_directory": str(settings.BASE_DIR)
                }
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
                    "parent": str(resolved_path.parent)
                }
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get file info: {e}"}
    
    def copy_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Copy a file from source to destination."""
        try:
            source_path = self._resolve_path(source)
            dest_path = self._resolve_path(destination)
            
            if not source_path.exists():
                return {"success": False, "error": f"Source file not found: {source_path}"}
            
            # Create destination directory if it doesn't exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(source_path, dest_path)
            return {"success": True, "output": f"File copied from {source_path} to {dest_path}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to copy file: {e}"}
    
    def move_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Move a file from source to destination."""
        try:
            source_path = self._resolve_path(source)
            dest_path = self._resolve_path(destination)
            
            if not source_path.exists():
                return {"success": False, "error": f"Source file not found: {source_path}"}
            
            # Create destination directory if it doesn't exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.move(str(source_path), str(dest_path))
            return {"success": True, "output": f"File moved from {source_path} to {dest_path}"}
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
                return {"success": True, "output": f"Directory deleted: {resolved_path}"}
            else:
                return {"success": False, "error": f"Unknown file type: {resolved_path}"}
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
                    "size_mb": round(total_size / 1024 / 1024, 2)
                }
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get directory size: {e}"}
    
    def find_large_files(self, directory: str, min_size_mb: float = 10.0) -> Dict[str, Any]:
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
                        large_files.append({
                            "path": str(item),
                            "size_mb": round(size / 1024 / 1024, 2)
                        })
            
            return {
                "success": True,
                "output": {
                    "large_files": large_files,
                    "count": len(large_files),
                    "min_size_mb": min_size_mb
                }
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
                    "usage_percent": round((disk_usage.used / disk_usage.total) * 100, 2)
                }
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to get disk usage: {e}"}
    
    def get_process_info(self) -> Dict[str, Any]:
        """Get information about running processes."""
        try:
            import psutil
            
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            return {
                "success": True,
                "output": {
                    "processes": processes[:20],  # Limit to first 20 processes
                    "total_processes": len(processes)
                }
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
            username = os.getenv('USER') or os.getenv('USERNAME') or 'user'
            
            # Define OS-specific root paths
            if sys.platform == "win32":
                root_paths = [
                    f"C:\\Users\\{username}\\Pictures\\Screenshots",
                    f"C:\\Users\\{username}\\Desktop\\Screenshots",
                    f"C:\\Users\\{username}\\Pictures",
                    f"C:\\Users\\{username}\\Desktop"
                ]
            elif sys.platform == "darwin":  # macOS
                root_paths = [
                    f"/Users/{username}/Pictures/Screenshots",
                    f"/Users/{username}/Desktop/Screenshots",
                    f"/Users/{username}/Pictures",
                    f"/Users/{username}/Desktop"
                ]
            else:  # Linux
                root_paths = [
                    f"/home/{username}/Pictures/Screenshots",
                    f"/home/{username}/Desktop/screenshots",
                    f"/home/{username}/Pictures",
                    f"/home/{username}/Desktop"
                ]
            
            # Try each path
            for path in root_paths:
                if os.path.exists(path):
                    # Try to navigate to it
                    change_result = self.change_dir(path)
                    if change_result.get("success"):
                        return {
                            "success": True,
                            "output": f"Successfully navigated to: {path}",
                            "path": path
                        }
            
            return {"success": False, "error": "No common user directories found"}
        except Exception as e:
            return {"success": False, "error": f"Failed to navigate to user directories: {e}"}
    
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
    
    def search_in_file(self, file_path: str, search_term: str, case_sensitive: bool = False) -> Dict[str, Any]:
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
                    matches.append({"position": pos, "line": content[:pos].count('\n') + 1})
                    start = pos + 1
            else:
                start = 0
                while True:
                    pos = content.find(search_term, start)
                    if pos == -1:
                        break
                    matches.append({"position": pos, "line": content[:pos].count('\n') + 1})
                    start = pos + 1
            
            return {"success": True, "matches": matches, "match_count": len(matches)}
        except Exception as e:
            return {"success": False, "error": f"Error searching in file: {e}"}

    def replace_in_file(self, file_path: str, old_text: str, new_text: str, case_sensitive: bool = True) -> Dict[str, Any]:
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
                return {"success": True, "message": "No replacements made.", "replacements": 0}

            replacements = content.count(old_text) if case_sensitive else len(re.findall(old_text, content, flags=re.IGNORECASE))
            
            resolved_path.write_text(new_content, encoding="utf-8")
            return {"success": True, "message": f"Replaced {replacements} occurrences.", "replacements": replacements}
        except Exception as e:
            return {"success": False, "error": f"Error replacing in file: {e}"}

    def search_directory(self, directory: str, search_term: str, case_sensitive: bool = False) -> Dict[str, Any]:
        """Recursively search for a term in all files within a directory."""
        try:
            resolved_path = self._resolve_path(directory)
            if not resolved_path.is_dir():
                return {"success": False, "error": f"Directory not found: {resolved_path}"}
            
            matches = []
            for root, _, files in os.walk(resolved_path):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        if not case_sensitive:
                            if search_term.lower() in content.lower():
                                matches.append(str(file_path.relative_to(resolved_path)))
                        else:
                            if search_term in content:
                                matches.append(str(file_path.relative_to(resolved_path)))
                    except Exception:
                        continue # Ignore files that can't be read
            
            return {"success": True, "matches": matches, "match_count": len(matches)}
        except Exception as e:
            return {"success": False, "error": f"Error searching directory: {e}"}

    def find_files(self, pattern: str, directory: str = ".") -> Dict[str, Any]:
        """Find files matching a glob pattern."""
        try:
            resolved_path = self._resolve_path(directory)
            if not resolved_path.is_dir():
                return {"success": False, "error": f"Directory not found: {resolved_path}"}
            
            found_files = [str(p.relative_to(resolved_path)) for p in resolved_path.glob(pattern)]
            return {"success": True, "files": found_files, "count": len(found_files)}
        except Exception as e:
            return {"success": False, "error": f"Error finding files: {e}"}

    def create_archive(self, archive_path: str, files: list, archive_type: str = "zip") -> Dict[str, Any]:
        """Create a zip or tar archive from a list of files."""
        try:
            import zipfile
            import tarfile
            
            resolved_archive_path = self._resolve_path(archive_path)
            resolved_archive_path.parent.mkdir(parents=True, exist_ok=True)
            
            resolved_files = [self._resolve_path(f) for f in files if self._resolve_path(f).exists()]
            if not resolved_files:
                return {"success": False, "error": "No valid files found to archive."}

            if archive_type.lower() == "zip":
                with zipfile.ZipFile(resolved_archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in resolved_files:
                        zipf.write(file_path, file_path.name)
            elif archive_type.lower() == "tar":
                with tarfile.open(resolved_archive_path, 'w') as tarf:
                    for file_path in resolved_files:
                        tarf.add(file_path, arcname=file_path.name)
            else:
                return {"success": False, "error": f"Unsupported archive type: {archive_type}"}
            
            return {"success": True, "message": f"Archive created: {resolved_archive_path}"}
        except Exception as e:
            return {"success": False, "error": f"Error creating archive: {e}"}

    def extract_archive(self, archive_path: str, extract_to: str = None) -> Dict[str, Any]:
        """Extract a zip or tar archive."""
        try:
            import zipfile
            import tarfile
            
            resolved_archive_path = self._resolve_path(archive_path)
            if not resolved_archive_path.is_file():
                return {"success": False, "error": f"Archive not found: {resolved_archive_path}"}

            extract_dir = self._resolve_path(extract_to) if extract_to else resolved_archive_path.parent / resolved_archive_path.stem
            extract_dir.mkdir(parents=True, exist_ok=True)

            if resolved_archive_path.suffix == '.zip':
                with zipfile.ZipFile(resolved_archive_path, 'r') as zipf:
                    zipf.extractall(extract_dir)
            elif '.tar' in resolved_archive_path.suffixes:
                with tarfile.open(resolved_archive_path, 'r:*') as tarf:
                    tarf.extractall(extract_dir)
            else:
                return {"success": False, "error": f"Unsupported archive format: {resolved_archive_path.name}"}

            return {"success": True, "message": f"Archive extracted to: {extract_dir}"}
        except Exception as e:
            return {"success": False, "error": f"Error extracting archive: {e}"}

    def read_json_file(self, file_path: str) -> Dict[str, Any]:
        """Read and parse a JSON file."""
        try:
            resolved_path = self._resolve_path(file_path)
            if not resolved_path.is_file():
                return {"success": False, "error": f"File not found: {resolved_path}"}
            
            with open(resolved_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return {"success": True, "data": data}
        except Exception as e:
            return {"success": False, "error": f"Error reading JSON file: {e}"}

    def write_json_file(self, file_path: str, data: Any, indent: int = 2) -> Dict[str, Any]:
        """Write data to a JSON file."""
        try:
            resolved_path = self._resolve_path(file_path)
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved_path, 'w', encoding='utf-8') as f:
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
            
            with open(resolved_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=delimiter)
                rows = list(reader)
            
            return {"success": True, "rows": rows}
        except Exception as e:
            return {"success": False, "error": f"Error reading CSV file: {e}"}

    def write_csv_file(self, file_path: str, data: list, headers: list = None, delimiter: str = ",") -> Dict[str, Any]:
        """Write data to a CSV file."""
        try:
            import csv
            
            resolved_path = self._resolve_path(file_path)
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved_path, 'w', encoding='utf-8', newline='') as f:
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
                return {"success": False, "error": "Pylint is not installed and installation failed."}
            # After successful installation, try to import again
            try:
                import importlib
                importlib.invalidate_caches() # Invalidate cache to ensure fresh import
                import pylint.lint
            except ImportError as e:
                return {"success": False, "error": f"Pylint installed but import failed: {e}"}

        resolved_path = self._resolve_path(path)
        
        # Capture stdout and stderr using subprocess instead of pylint's Run class
        try:
            import subprocess
            result = subprocess.run(
                ["pylint", str(resolved_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "success": True, 
                "stdout": result.stdout, 
                "stderr": result.stderr,
                "return_code": result.returncode,
                "output": f"Linter output:\n{result.stdout}\n{result.stderr}" if result.stdout or result.stderr else "No linting issues found"
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Pylint execution timed out"}
        except Exception as e:
            return {"success": False, "error": f"Error running pylint: {e}"}

    def replace_in_multiple_files(self, files: list, old_text: str, new_text: str, case_sensitive: bool = True) -> Dict[str, Any]:
        """Perform a search and replace operation across multiple files."""
        results = []
        for file_path in files:
            result = self.replace_in_file(file_path, old_text, new_text, case_sensitive)
            results.append({"file_path": file_path, "result": result})
        
        return {"success": True, "results": results}
    
    def click_screen(self, x: int, y: int, button: str = "left") -> Dict[str, Any]:
        """Click at specific screen coordinates using xdotool."""
        try:
            import subprocess
            
            # Use xdotool to click at coordinates
            cmd = ["xdotool", "click", "--clearmodifiers", str(x), str(y)]
            if button != "left":
                cmd.extend(["--button", button])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "output": f"Clicked at coordinates ({x}, {y}) with {button} button",
                    "message": f"Successfully clicked at ({x}, {y})"
                }
            else:
                return {
                    "success": False,
                    "error": f"xdotool failed: {result.stderr}",
                    "output": f"Failed to click at ({x}, {y})"
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Click operation timed out"}
        except Exception as e:
            return {"success": False, "error": f"Error clicking screen: {e}"}
    
    def get_mouse_position(self) -> Dict[str, Any]:
        """Get current mouse position using xdotool."""
        try:
            import subprocess
            
            result = subprocess.run(["xdotool", "getmouselocation", "--shell"], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Parse the output
                lines = result.stdout.strip().split('\n')
                coords = {}
                for line in lines:
                    if '=' in line:
                        key, value = line.split('=', 1)
                        coords[key] = int(value)
                
                return {
                    "success": True,
                    "output": f"Mouse position: X={coords.get('X', 0)}, Y={coords.get('Y', 0)}",
                    "x": coords.get('X', 0),
                    "y": coords.get('Y', 0),
                    "message": f"Mouse is at ({coords.get('X', 0)}, {coords.get('Y', 0)})"
                }
            else:
                return {
                    "success": False,
                    "error": f"xdotool failed: {result.stderr}",
                    "output": "Failed to get mouse position"
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Mouse position query timed out"}
        except Exception as e:
            return {"success": False, "error": f"Error getting mouse position: {e}"}
    
    def read_screen(self, prompt: str = "Describe what you see on this screen in detail") -> Dict[str, Any]:
        """Capture the current screen, analyze it with AI, and clean up the temporary image."""
        try:
            import tempfile
            import os
            from src.core.gemini_client import GeminiClient
            
            # Create temporary file for screenshot
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                screenshot_path = temp_file.name
            
            # Take screenshot using system command
            if os.name == 'nt':  # Windows
                # Use Windows Snipping Tool or PowerShell
                screenshot_cmd = f'powershell -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Screen]::PrimaryScreen.Bounds | ForEach-Object {{ $bounds = $_; $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height; $graphics = [System.Drawing.Graphics]::FromImage($bitmap); $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size); $bitmap.Save(\'{screenshot_path}\', [System.Drawing.Imaging.ImageFormat]::Png); $bitmap.Dispose(); $graphics.Dispose() }}"'
            else:  # Linux/macOS
                # Use scrot (Linux) or screencapture (macOS)
                if os.path.exists('/usr/bin/scrot'):
                    screenshot_cmd = f'scrot "{screenshot_path}"'
                elif os.path.exists('/usr/bin/gnome-screenshot'):
                    screenshot_cmd = f'gnome-screenshot -f "{screenshot_path}"'
                elif os.path.exists('/usr/bin/import'):
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
                    "output": "Screenshot capture failed"
                }
            
            # Check if screenshot file was created and has content
            if not os.path.exists(screenshot_path) or os.path.getsize(screenshot_path) == 0:
                return {
                    "success": False,
                    "error": "Screenshot file was not created or is empty",
                    "output": "Screenshot capture failed - no image data"
                }
            
            # Initialize Gemini client and analyze the screenshot
            gemini_client = GeminiClient()
            analysis_result = gemini_client.analyze_image(screenshot_path, prompt)
            
            # Clean up temporary screenshot file
            try:
                os.unlink(screenshot_path)
            except Exception as cleanup_error:
                self.logger.warning(f"Failed to clean up temporary screenshot: {cleanup_error}")
            
            if analysis_result:
                # Format the analysis result for better readability
                formatted_output = self._format_screen_analysis(analysis_result)
                
                return {
                    "success": True,
                    "text": analysis_result,
                    "output": formatted_output,
                    "message": "Screen captured and analyzed successfully"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to analyze screenshot",
                    "output": "No analysis result returned"
                }
                
        except Exception as e:
            # Clean up temporary file if it exists
            try:
                if 'screenshot_path' in locals() and os.path.exists(screenshot_path):
                    os.unlink(screenshot_path)
            except:
                pass
            
            return {
                "success": False,
                "error": f"Error reading screen: {e}",
                "output": f"Error: {e}"
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
                        "output": f"Please install {package_name} manually. Supported managers: apt, yum, dnf, pacman, zypper"
                    }
            elif system == "darwin":  # macOS
                if os.path.exists("/usr/local/bin/brew"):
                    cmd = f"brew install {package_name}"
                    manager = "homebrew"
                else:
                    return {
                        "success": False,
                        "error": "Homebrew not found",
                        "output": f"Please install Homebrew first: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\" then run: brew install {package_name}"
                    }
            elif system == "windows":
                if os.path.exists("C:\\ProgramData\\chocolatey\\bin\\choco.exe"):
                    cmd = f"choco install {package_name} -y"
                    manager = "chocolatey"
                else:
                    return {
                        "success": False,
                        "error": "Chocolatey not found",
                        "output": f"Please install Chocolatey first: https://chocolatey.org/install then run: choco install {package_name}"
                    }
            else:
                return {
                    "success": False,
                    "error": f"Unsupported operating system: {system}",
                    "output": f"Please install {package_name} manually for {system}"
                }

            result = self.run_shell(cmd)

            if result["success"]:
                return {
                    "success": True,
                    "output": f"Package '{package_name}' installed successfully using {manager}",
                    "message": f"Successfully installed {package_name} using {manager}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to install {package_name} using {manager}",
                    "output": f"Installation failed: {result.get('error', 'Unknown error')}. Manual installation required.",
                    "manual_command": cmd
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error installing system package: {e}",
                "output": f"Please install {package_name} manually"
            }

    def check_system_dependency(self, dependency: str) -> Dict[str, Any]:
        """Check if a system dependency is installed and provide installation instructions if not."""
        try:
            import subprocess
            import platform
            import os

            result = subprocess.run(["which", dependency], capture_output=True, text=True)

            if result.returncode == 0:
                version_result = subprocess.run([dependency, "--version"], capture_output=True, text=True)
                version = version_result.stdout.strip() if version_result.returncode == 0 else "Unknown"

                return {
                    "success": True,
                    "installed": True,
                    "path": result.stdout.strip(),
                    "version": version,
                    "output": f"{dependency} is installed at {result.stdout.strip()}"
                }
            else:
                system = platform.system().lower()
                instructions = self._get_installation_instructions(dependency, system)

                return {
                    "success": False,
                    "installed": False,
                    "error": f"{dependency} is not installed",
                    "output": f"{dependency} not found. Installation instructions: {instructions}",
                    "instructions": instructions
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error checking dependency: {e}",
                "output": f"Could not check if {dependency} is installed"
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
                "pyautogui": "pip install pyautogui (in virtual environment)"
            },
            "darwin": {
                "screencapture": "Built-in on macOS",
                "xdotool": "brew install xdotool",
                "pylint": "pip install pylint (in virtual environment)",
                "pyautogui": "pip install pyautogui (in virtual environment)"
            },
            "windows": {
                "xdotool": "Not available on Windows. Use pyautogui instead.",
                "pylint": "pip install pylint (in virtual environment)",
                "pyautogui": "pip install pyautogui (in virtual environment)"
            }
        }

        return instructions.get(system, {}).get(dependency, f"Please install {dependency} manually for {system}")
    
    def _format_screen_analysis(self, analysis_text: str) -> str:
        """Format screen analysis text for better readability."""
        try:
            # Split the analysis into sections
            lines = analysis_text.split('\n')
            formatted_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    formatted_lines.append("")
                    continue
                
                # Format headers and subheaders
                if line.startswith('**') and line.endswith('**'):
                    # Main header
                    header_text = line[2:-2]
                    formatted_lines.append(f"\n{'='*60}")
                    formatted_lines.append(f"📋 {header_text.upper()}")
                    formatted_lines.append(f"{'='*60}")
                elif line.startswith('*   **') and line.endswith('**'):
                    # Subheader
                    subheader_text = line[6:-2]
                    formatted_lines.append(f"\n🔹 {subheader_text}")
                    formatted_lines.append("-" * 40)
                elif line.startswith('*   '):
                    # Bullet point
                    bullet_text = line[4:]
                    formatted_lines.append(f"  • {bullet_text}")
                elif line.startswith('1.  **') and line.endswith('**'):
                    # Numbered item with bold
                    item_text = line[6:-2]
                    formatted_lines.append(f"\n📌 {item_text}")
                elif line.startswith('2.  **') and line.endswith('**'):
                    # Numbered item with bold
                    item_text = line[6:-2]
                    formatted_lines.append(f"\n📌 {item_text}")
                elif line.startswith('3.  **') and line.endswith('**'):
                    # Numbered item with bold
                    item_text = line[6:-2]
                    formatted_lines.append(f"\n📌 {item_text}")
                else:
                    # Regular text
                    formatted_lines.append(f"  {line}")
            
            # Join all lines and add some spacing
            formatted_text = "\n".join(formatted_lines)
            
            # Add a header and footer
            final_output = f"""
🖥️  SCREEN ANALYSIS REPORT
{'='*60}

{formatted_text}

{'='*60}
✅ Analysis completed successfully
"""
            
            return final_output.strip()
            
        except Exception as e:
            # If formatting fails, return the original text with a simple header
            return f"""
🖥️  SCREEN ANALYSIS
{'='*50}

{analysis_text}

{'='*50}
✅ Analysis completed
"""
