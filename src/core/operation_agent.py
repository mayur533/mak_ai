#!/usr/bin/env python3
"""
OperationAgent - Sandboxed file and process operations.
The only agent that directly interacts with the operating system.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import aiofiles
import psutil
from dataclasses import dataclass
from enum import Enum

from .message_bus import MessageBus, MessageEnvelope, MessageType, ChannelType
from .gemini_client import get_gemini_client

logger = logging.getLogger(__name__)

class OperationType(Enum):
    """Types of operations supported by OperationAgent."""
    PROCESS = "process"
    FS = "fs"

class FileOperation(Enum):
    """File system operations."""
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    APPEND_FILE = "append_file"
    REPLACE_IN_FILE = "replace_in_file"
    LIST_DIR = "list_dir"
    GLOB = "glob"
    MAKE_DIR = "make_dir"
    REMOVE_FILE = "remove_file"
    COPY_FILE = "copy_file"
    MOVE_FILE = "move_file"

@dataclass
class OperationRequest:
    """Request for an operation."""
    op_kind: OperationType
    op: Optional[str] = None  # For file operations
    cmd: Optional[List[str]] = None  # For process operations
    path: Optional[str] = None
    content: Optional[str] = None
    find: Optional[str] = None
    replace: Optional[str] = None
    pattern: Optional[str] = None
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    timeout_secs: int = 30
    permit: bool = False

@dataclass
class OperationResponse:
    """Response from an operation."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_secs: float = 0.0
    exit_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None

class SandboxManager:
    """Manages sandbox environment for safe operations."""
    
    def __init__(self, base_path: str, allowed_paths: List[str], max_processes: int = 5):
        self.base_path = Path(base_path)
        self.allowed_paths = [Path(p) for p in allowed_paths]
        self.max_processes = max_processes
        self.running_processes: Dict[str, subprocess.Popen] = {}
        
        # Create sandbox directory
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Allowed commands
        self.allowed_commands = {
            "python", "python3", "node", "npm", "pip", "git", 
            "ls", "cat", "echo", "find", "grep", "awk", "sed", 
            "curl", "wget", "mkdir", "cp", "mv", "rm", "touch"
        }
        
        # Allowed file extensions
        self.allowed_extensions = {
            ".txt", ".json", ".yaml", ".yml", ".py", ".md", 
            ".log", ".csv", ".xml", ".html", ".css", ".js"
        }
    
    def is_path_allowed(self, path: str) -> bool:
        """Check if a path is within allowed directories."""
        try:
            path_obj = Path(path).resolve()
            return any(path_obj.is_relative_to(allowed) for allowed in self.allowed_paths)
        except Exception:
            return False
    
    def is_command_allowed(self, cmd: List[str]) -> bool:
        """Check if a command is allowed."""
        if not cmd:
            return False
        return cmd[0] in self.allowed_commands
    
    def sanitize_path(self, path: str) -> str:
        """Sanitize and validate a file path."""
        path_obj = Path(path)
        
        # Resolve relative paths
        if not path_obj.is_absolute():
            path_obj = self.base_path / path_obj
        
        # Check if within allowed paths
        if not self.is_path_allowed(str(path_obj)):
            raise PermissionError(f"Path {path} is not within allowed directories")
        
        return str(path_obj)
    
    def check_file_extension(self, path: str) -> bool:
        """Check if file extension is allowed."""
        ext = Path(path).suffix.lower()
        return ext in self.allowed_extensions or ext == ""
    
    def get_safe_environment(self, env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get a safe environment for process execution."""
        safe_env = os.environ.copy()
        
        # Remove dangerous environment variables
        dangerous_vars = {
            "PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "NODE_PATH",
            "HOME", "USER", "USERNAME", "SHELL"
        }
        
        for var in dangerous_vars:
            safe_env.pop(var, None)
        
        # Add custom environment variables
        if env:
            safe_env.update(env)
        
        return safe_env

class OperationAgent:
    """Agent responsible for safe file and process operations."""
    
    def __init__(self, message_bus: MessageBus, sandbox_config: Dict[str, Any]):
        self.message_bus = message_bus
        self.agent_id = "operation.agent"
        self.running = False
        
        # Initialize sandbox
        self.sandbox = SandboxManager(
            base_path=sandbox_config["base_path"],
            allowed_paths=sandbox_config["allowed_paths"],
            max_processes=sandbox_config["max_processes"]
        )
        
        # Operation limits
        self.max_file_size = sandbox_config.get("max_file_size_mb", 100) * 1024 * 1024
        self.max_process_timeout = sandbox_config.get("max_process_timeout", 60)
        self.max_process_memory = sandbox_config.get("max_process_memory_mb", 512) * 1024 * 1024
        
        # Statistics
        self.stats = {
            "operations_performed": 0,
            "operations_failed": 0,
            "files_processed": 0,
            "processes_executed": 0,
            "start_time": datetime.now()
        }
    
    async def start(self):
        """Start the OperationAgent."""
        self.running = True
        
        # Subscribe to operation requests
        self.message_bus.subscribe(
            ChannelType.OPERATION_AGENT.value,
            self._handle_operation_request
        )
        
        # Start heartbeat
        asyncio.create_task(self._send_heartbeat())
        
        logger.info("OperationAgent started")
    
    async def stop(self):
        """Stop the OperationAgent."""
        self.running = False
        
        # Kill all running processes
        for process_id, process in self.running_processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Failed to terminate process {process_id}: {e}")
        
        logger.info("OperationAgent stopped")
    
    async def _send_heartbeat(self):
        """Send periodic heartbeat."""
        while self.running:
            try:
                heartbeat = MessageEnvelope(
                    id=str(uuid.uuid4()),
                    from_agent=self.agent_id,
                    to="*",
                    type=MessageType.HEARTBEAT,
                    timestamp=datetime.now().isoformat(),
                    payload={
                        "status": "running",
                        "stats": self.stats,
                        "running_processes": len(self.running_processes)
                    }
                )
                
                await self.message_bus.publish(heartbeat)
                await asyncio.sleep(5)  # Send heartbeat every 5 seconds
                
            except Exception as e:
                logger.error(f"Failed to send heartbeat: {e}")
                await asyncio.sleep(5)
    
    async def _handle_operation_request(self, message: MessageEnvelope):
        """Handle an operation request."""
        try:
            if message.type != MessageType.OPERATION_REQUEST:
                return
            
            # Parse operation request
            request_data = message.payload
            operation = OperationRequest(**request_data)
            
            # Execute operation
            response = await self._execute_operation(operation)
            
            # Send response
            response_message = MessageEnvelope(
                id=str(uuid.uuid4()),
                from_agent=self.agent_id,
                to=message.from_agent,
                type=MessageType.OPERATION_RESPONSE,
                timestamp=datetime.now().isoformat(),
                reply_to=message.id,
                payload=asdict(response)
            )
            
            await self.message_bus.publish(response_message)
            
        except Exception as e:
            logger.error(f"Error handling operation request: {e}")
            
            # Send error response
            error_response = MessageEnvelope(
                id=str(uuid.uuid4()),
                from_agent=self.agent_id,
                to=message.from_agent,
                type=MessageType.OPERATION_RESPONSE,
                timestamp=datetime.now().isoformat(),
                reply_to=message.id,
                payload={
                    "success": False,
                    "error": str(e),
                    "duration_secs": 0.0
                }
            )
            
            await self.message_bus.publish(error_response)
    
    async def _execute_operation(self, operation: OperationRequest) -> OperationResponse:
        """Execute an operation safely."""
        start_time = time.time()
        
        try:
            if operation.op_kind == OperationType.PROCESS:
                result = await self._execute_process(operation)
            elif operation.op_kind == OperationType.FS:
                result = await self._execute_file_operation(operation)
            else:
                raise ValueError(f"Unknown operation type: {operation.op_kind}")
            
            duration = time.time() - start_time
            self.stats["operations_performed"] += 1
            
            return OperationResponse(
                success=True,
                result=result,
                duration_secs=duration
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.stats["operations_failed"] += 1
            logger.error(f"Operation failed: {e}")
            
            return OperationResponse(
                success=False,
                error=str(e),
                duration_secs=duration
            )
    
    async def _execute_process(self, operation: OperationRequest) -> Dict[str, Any]:
        """Execute a process operation."""
        if not operation.cmd:
            raise ValueError("Command is required for process operations")
        
        # Validate command
        if not self.sandbox.is_command_allowed(operation.cmd):
            raise PermissionError(f"Command {operation.cmd[0]} is not allowed")
        
        # Set working directory
        cwd = operation.cwd
        if cwd:
            cwd = self.sandbox.sanitize_path(cwd)
        else:
            cwd = str(self.sandbox.base_path)
        
        # Prepare environment
        env = self.sandbox.get_safe_environment(operation.env)
        
        # Execute process
        process_id = str(uuid.uuid4())
        
        try:
            process = await asyncio.create_subprocess_exec(
                *operation.cmd,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.running_processes[process_id] = process
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=operation.timeout_secs
                )
                
                self.stats["processes_executed"] += 1
                
                return {
                    "exit_code": process.returncode,
                    "stdout": stdout.decode('utf-8', errors='replace'),
                    "stderr": stderr.decode('utf-8', errors='replace'),
                    "process_id": process_id
                }
                
            except asyncio.TimeoutError:
                process.kill()
                raise TimeoutError(f"Process timed out after {operation.timeout_secs} seconds")
                
        finally:
            self.running_processes.pop(process_id, None)
    
    async def _execute_file_operation(self, operation: OperationRequest) -> Any:
        """Execute a file system operation."""
        if not operation.op:
            raise ValueError("Operation type is required for file operations")
        
        op_type = FileOperation(operation.op)
        
        if op_type == FileOperation.READ_FILE:
            return await self._read_file(operation.path)
        elif op_type == FileOperation.WRITE_FILE:
            return await self._write_file(operation.path, operation.content, operation.permit)
        elif op_type == FileOperation.APPEND_FILE:
            return await self._append_file(operation.path, operation.content, operation.permit)
        elif op_type == FileOperation.REPLACE_IN_FILE:
            return await self._replace_in_file(operation.path, operation.find, operation.replace, operation.permit)
        elif op_type == FileOperation.LIST_DIR:
            return await self._list_dir(operation.path)
        elif op_type == FileOperation.GLOB:
            return await self._glob(operation.pattern)
        elif op_type == FileOperation.MAKE_DIR:
            return await self._make_dir(operation.path, operation.permit)
        elif op_type == FileOperation.REMOVE_FILE:
            return await self._remove_file(operation.path, operation.permit)
        elif op_type == FileOperation.COPY_FILE:
            return await self._copy_file(operation.path, operation.content, operation.permit)
        elif op_type == FileOperation.MOVE_FILE:
            return await self._move_file(operation.path, operation.content, operation.permit)
        else:
            raise ValueError(f"Unknown file operation: {operation.op}")
    
    async def _read_file(self, path: str) -> str:
        """Read a file safely."""
        if not path:
            raise ValueError("Path is required")
        
        safe_path = self.sandbox.sanitize_path(path)
        
        if not self.sandbox.check_file_extension(safe_path):
            raise PermissionError(f"File extension not allowed: {Path(safe_path).suffix}")
        
        # Check file size
        file_size = os.path.getsize(safe_path)
        if file_size > self.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
        
        async with aiofiles.open(safe_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        self.stats["files_processed"] += 1
        return content
    
    async def _write_file(self, path: str, content: str, permit: bool = False) -> Dict[str, Any]:
        """Write a file safely."""
        if not path:
            raise ValueError("Path is required")
        
        if not permit:
            raise PermissionError("Write operation requires explicit permit")
        
        safe_path = self.sandbox.sanitize_path(path)
        
        if not self.sandbox.check_file_extension(safe_path):
            raise PermissionError(f"File extension not allowed: {Path(safe_path).suffix}")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        
        async with aiofiles.open(safe_path, 'w', encoding='utf-8') as f:
            await f.write(content)
        
        self.stats["files_processed"] += 1
        return {"path": safe_path, "size": len(content)}
    
    async def _append_file(self, path: str, content: str, permit: bool = False) -> Dict[str, Any]:
        """Append to a file safely."""
        if not path:
            raise ValueError("Path is required")
        
        if not permit:
            raise PermissionError("Append operation requires explicit permit")
        
        safe_path = self.sandbox.sanitize_path(path)
        
        async with aiofiles.open(safe_path, 'a', encoding='utf-8') as f:
            await f.write(content)
        
        self.stats["files_processed"] += 1
        return {"path": safe_path, "appended": len(content)}
    
    async def _replace_in_file(self, path: str, find: str, replace: str, permit: bool = False) -> Dict[str, Any]:
        """Replace text in a file safely."""
        if not path or not find:
            raise ValueError("Path and find text are required")
        
        if not permit:
            raise PermissionError("Replace operation requires explicit permit")
        
        safe_path = self.sandbox.sanitize_path(path)
        
        # Read file
        async with aiofiles.open(safe_path, 'r', encoding='utf-8') as f:
            content = await f.read()
        
        # Replace text
        new_content = content.replace(find, replace)
        replacements = content.count(find)
        
        # Write back
        async with aiofiles.open(safe_path, 'w', encoding='utf-8') as f:
            await f.write(new_content)
        
        self.stats["files_processed"] += 1
        return {"path": safe_path, "replacements": replacements}
    
    async def _list_dir(self, path: str) -> List[Dict[str, Any]]:
        """List directory contents safely."""
        if not path:
            path = str(self.sandbox.base_path)
        
        safe_path = self.sandbox.sanitize_path(path)
        
        items = []
        for item in os.listdir(safe_path):
            item_path = os.path.join(safe_path, item)
            stat = os.stat(item_path)
            
            items.append({
                "name": item,
                "path": item_path,
                "is_file": os.path.isfile(item_path),
                "is_dir": os.path.isdir(item_path),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return items
    
    async def _glob(self, pattern: str) -> List[str]:
        """Glob pattern matching safely."""
        if not pattern:
            raise ValueError("Pattern is required")
        
        # Ensure pattern is within allowed paths
        pattern_path = Path(pattern)
        if not any(pattern_path.is_relative_to(allowed) for allowed in self.sandbox.allowed_paths):
            # Make pattern relative to sandbox
            pattern = str(self.sandbox.base_path / pattern)
        
        import glob
        matches = glob.glob(pattern)
        
        # Filter to only allowed paths
        safe_matches = []
        for match in matches:
            if self.sandbox.is_path_allowed(match):
                safe_matches.append(match)
        
        return safe_matches
    
    async def _make_dir(self, path: str, permit: bool = False) -> Dict[str, Any]:
        """Create a directory safely."""
        if not path:
            raise ValueError("Path is required")
        
        if not permit:
            raise PermissionError("Make directory operation requires explicit permit")
        
        safe_path = self.sandbox.sanitize_path(path)
        os.makedirs(safe_path, exist_ok=True)
        
        return {"path": safe_path, "created": True}
    
    async def _remove_file(self, path: str, permit: bool = False) -> Dict[str, Any]:
        """Remove a file safely."""
        if not path:
            raise ValueError("Path is required")
        
        if not permit:
            raise PermissionError("Remove operation requires explicit permit")
        
        safe_path = self.sandbox.sanitize_path(path)
        
        if os.path.isfile(safe_path):
            os.remove(safe_path)
            return {"path": safe_path, "removed": True}
        else:
            return {"path": safe_path, "removed": False, "error": "File not found"}
    
    async def _copy_file(self, src: str, dst: str, permit: bool = False) -> Dict[str, Any]:
        """Copy a file safely."""
        if not src or not dst:
            raise ValueError("Source and destination paths are required")
        
        if not permit:
            raise PermissionError("Copy operation requires explicit permit")
        
        safe_src = self.sandbox.sanitize_path(src)
        safe_dst = self.sandbox.sanitize_path(dst)
        
        import shutil
        shutil.copy2(safe_src, safe_dst)
        
        return {"src": safe_src, "dst": safe_dst, "copied": True}
    
    async def _move_file(self, src: str, dst: str, permit: bool = False) -> Dict[str, Any]:
        """Move a file safely."""
        if not src or not dst:
            raise ValueError("Source and destination paths are required")
        
        if not permit:
            raise PermissionError("Move operation requires explicit permit")
        
        safe_src = self.sandbox.sanitize_path(src)
        safe_dst = self.sandbox.sanitize_path(dst)
        
        import shutil
        shutil.move(safe_src, safe_dst)
        
        return {"src": safe_src, "dst": safe_dst, "moved": True}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get operation statistics."""
        uptime = (datetime.now() - self.stats["start_time"]).total_seconds()
        return {
            **self.stats,
            "uptime_seconds": uptime,
            "operations_per_second": self.stats["operations_performed"] / max(uptime, 1),
            "success_rate": self.stats["operations_performed"] / max(
                self.stats["operations_performed"] + self.stats["operations_failed"], 1
            )
        }
