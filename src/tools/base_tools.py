"""
Base tools for the AI Assistant System.
Contains core functionality tools that are always available.
"""

import ast
import io
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
        Execute a shell command.
        
        Args:
            command: The shell command to execute
            
        Returns:
            Dict with success status and output/error
        """
        self.logger.info(f"Executing shell command: `{command}`")
        try:
            # Show command execution in real-time
            print(f"\n🔧 Executing: {command}")
            print("─" * 50)
            
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                timeout=300,
                cwd=settings.BASE_DIR
            )
            
            print("─" * 50)
            
            # Capture output for return value
            captured_result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=settings.BASE_DIR
            )
            
            output = f"stdout:\n{captured_result.stdout}\nstderr:\n{captured_result.stderr}"
            if captured_result.returncode == 0:
                self.logger.success(f"Command completed successfully")
                return {"success": True, "output": output}
            else:
                self.logger.error(f"Command failed with return code {captured_result.returncode}")
                return {"success": False, "error": output}
        except subprocess.TimeoutExpired:
            error_msg = "Command timed out after 300 seconds"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {"success": False, "error": error_msg}
    
    def run_python(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code with access to all system tools.
        
        Args:
            code: The Python code to execute
            
        Returns:
            Dict with success status and output/error
        """
        self.logger.info(f"Executing Python code:\n{code}")
        
        # Create execution context with system tools
        exec_context = {
            "system": self.system,
            "os": os,
            "Path": Path,
            "settings": settings,
            "logger": logger
        }
        
        # Add all available tools to context
        if self.system and hasattr(self.system, 'tools'):
            for tool_name, tool in self.system.tools.items():
                if tool.func:
                    exec_context[tool_name] = tool.func
        
        # Capture stdout and stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        redirected_output = io.StringIO()
        sys.stdout, sys.stderr = redirected_output, redirected_output
        
        try:
            exec(code, globals(), exec_context)
            output = redirected_output.getvalue()
            return {"success": True, "output": output}
        except Exception as e:
            error_msg = f"Python execution error: {str(e)}\n{traceback.format_exc()}"
            return {"success": False, "error": error_msg}
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
    
    def install_package(self, package_name: str) -> Dict[str, Any]:
        """
        Install a Python package using pip.
        
        Args:
            package_name: The name of the package to install
            
        Returns:
            Dict with success status and output/error
        """
        self.logger.info(f"Installing package: {package_name}")
        return self.run_shell(f"python3 -m pip install {package_name}")
    
    def read_file(self, file_path: str) -> Dict[str, Any]:
        """
        Read the content of a file.
        
        Args:
            file_path: The path to the file to read
            
        Returns:
            Dict with success status and content/error
        """
        try:
            path = Path(file_path)
            if not path.is_file():
                return {"success": False, "error": f"File not found: {file_path}"}
            
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            return {"success": True, "output": content}
        except Exception as e:
            return {"success": False, "error": f"Failed to read file: {e}"}
    
    def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Write content to a file.
        
        Args:
            file_path: The path to the file to write
            content: The content to write
            
        Returns:
            Dict with success status and message/error
        """
        try:
            path = Path(file_path)
            if not path.is_absolute():
                path = settings.OUTPUT_DIR / path
            
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return {"success": True, "output": f"Successfully wrote content to {path}"}
        except Exception as e:
            return {"success": False, "error": f"Failed to write file: {e}"}
    
    def list_dir(self, directory: str = ".") -> Dict[str, Any]:
        """
        List the contents of a directory.
        
        Args:
            directory: The path to the directory to list
            
        Returns:
            Dict with success status and directory contents/error
        """
        try:
            path = Path(directory)
            if not path.is_dir():
                return {"success": False, "error": f"Directory not found: {directory}"}
            
            contents = [str(item) for item in path.iterdir()]
            return {"success": True, "output": "\n".join(contents)}
        except Exception as e:
            return {"success": False, "error": f"Failed to list directory: {e}"}
    
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
            
            # Create execution context
            exec_context = {"system": self.system}
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
