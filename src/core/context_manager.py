"""
Advanced Context Management System for AI Assistant.
Handles persistent context, project tracking, and context caching with summarization.
"""

import json
import os
import time
import gzip
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config.settings import settings
from src.logging.logger import logger


@dataclass
class ProjectContext:
    """Represents a project context with all relevant information."""
    project_id: str
    project_name: str
    project_path: str
    created_at: float
    last_accessed: float
    current_directory: str
    full_context: List[Dict[str, Any]]  # Complete history from start to finish (never summarized)
    actual_context: List[Dict[str, Any]]  # Current working context (gets summarized when full)
    context_summary: str  # Summarized version of actual_context when it gets full
    context_hash: str
    is_active: bool = True


@dataclass
class ContextEntry:
    """Represents a single context entry."""
    timestamp: float
    entry_type: str  # 'action', 'result', 'error', 'info'
    content: str
    metadata: Dict[str, Any]
    entry_id: str


class ContextManager:
    """Manages persistent context across sessions with caching and summarization."""
    
    def __init__(self):
        """Initialize the context manager."""
        self.logger = logger
        self.context_file = settings.BASE_DIR / "db" / "context.json"
        self.system_file = settings.BASE_DIR / "system.json"
        self.max_context_tokens = 10000  # Token limit for context
        self.max_context_entries = 1000  # Fallback entry limit
        self.current_project: Optional[ProjectContext] = None
        self.projects: Dict[str, ProjectContext] = {}
        self.system_config = {}
        
        # Ensure db directory exists
        self.context_file.parent.mkdir(exist_ok=True)
        
        # Compression settings
        self.compression_enabled = True
        self.compression_level = 6
        self.min_compress_size = 1024
        
        # Context caching
        self._context_cache = {}
        self._cache_lock = threading.RLock()
        
        self._load_system_config()
        self._load_context()
        self._load_projects()
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data for storage."""
        if not self.compression_enabled:
            return json.dumps(data, default=str).encode()
        
        try:
            json_str = json.dumps(data, default=str)
            if len(json_str.encode()) > self.min_compress_size:
                compressed = gzip.compress(json_str.encode(), compresslevel=self.compression_level)
                return b'COMPRESSED:' + compressed
            else:
                return b'UNCOMPRESSED:' + json_str.encode()
        except Exception as e:
            self.logger.error(f"Error compressing data: {e}")
            return json.dumps(data, default=str).encode()
    
    def _decompress_data(self, data: bytes) -> Any:
        """Decompress data from storage."""
        try:
            if data.startswith(b'COMPRESSED:'):
                compressed_data = data[11:]
                json_str = gzip.decompress(compressed_data).decode()
                return json.loads(json_str)
            elif data.startswith(b'UNCOMPRESSED:'):
                json_str = data[13:].decode()
                return json.loads(json_str)
            else:
                return json.loads(data.decode())
        except Exception as e:
            self.logger.error(f"Error decompressing data: {e}")
            return {}
    
    def _load_system_config(self):
        """Load system configuration from file."""
        if self.system_file.exists():
            try:
                with open(self.system_file, 'r', encoding='utf-8') as f:
                    self.system_config = json.load(f)
                self.logger.info("System config loaded from file")
            except Exception as e:
                self.logger.error(f"Failed to load system config: {e}")
                self.system_config = {}
    
    def _load_context(self):
        """Load context from file with compression support."""
        if self.context_file.exists():
            try:
                # Try to read as compressed data first
                with open(self.context_file, 'rb') as f:
                    data_bytes = f.read()
                
                if data_bytes.startswith(b'COMPRESSED:') or data_bytes.startswith(b'UNCOMPRESSED:'):
                    # Compressed format
                    data = self._decompress_data(data_bytes)
                else:
                    # Legacy JSON format
                    data = json.loads(data_bytes.decode())
                
                self.current_project = ProjectContext(**data) if data else None
                self.logger.info("Context loaded from file")
            except Exception as e:
                self.logger.error(f"Failed to load context: {e}")
                self.current_project = None
    
    def _load_projects(self):
        """Load projects from system file."""
        if self.system_file.exists():
            try:
                with open(self.system_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'projects' in data:
                        self.projects = {
                            pid: ProjectContext(**project_data) 
                            for pid, project_data in data['projects'].items()
                        }
                        self.logger.info(f"Loaded {len(self.projects)} projects from system file")
            except Exception as e:
                self.logger.error(f"Failed to load projects: {e}")
                self.projects = {}
    
    def _save_context(self):
        """Save current context to file with compression."""
        try:
            if self.current_project:
                # Compress data
                compressed_data = self._compress_data(asdict(self.current_project))
                
                # Save compressed data
                with open(self.context_file, 'wb') as f:
                    f.write(compressed_data)
                
                # Update cache
                with self._cache_lock:
                    self._context_cache['current_project'] = {
                        'data': self.current_project,
                        'timestamp': time.time()
                    }
        except Exception as e:
            self.logger.error(f"Failed to save context: {e}")
    
    def _save_projects(self):
        """Save all projects to system file."""
        try:
            # Load existing system config
            system_data = {}
            if self.system_file.exists():
                with open(self.system_file, 'r', encoding='utf-8') as f:
                    system_data = json.load(f)
            
            # Update projects section
            system_data['projects'] = {pid: asdict(project) for pid, project in self.projects.items()}
            
            # Save updated system config
            with open(self.system_file, 'w', encoding='utf-8') as f:
                json.dump(system_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save projects: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get context cache statistics."""
        with self._cache_lock:
            return {
                'cache_size': len(self._context_cache),
                'compression_enabled': self.compression_enabled,
                'compression_level': self.compression_level
            }
    
    def optimize_context_storage(self):
        """Optimize context storage by compressing and cleaning up."""
        try:
            if self.current_project:
                # Recompress with optimal settings
                compressed_data = self._compress_data(asdict(self.current_project))
                
                # Save optimized version
                with open(self.context_file, 'wb') as f:
                    f.write(compressed_data)
                
                self.logger.info("Context storage optimized")
        except Exception as e:
            self.logger.error(f"Error optimizing context storage: {e}")
    
    def clear_cache(self):
        """Clear context cache."""
        with self._cache_lock:
            self._context_cache.clear()
        self.logger.info("Context cache cleared")
    
    def get_system_directories(self) -> List[str]:
        """Get list of protected system directories."""
        return self.system_config.get('system_directories', [])
    
    def get_protected_files(self) -> List[str]:
        """Get list of protected files."""
        return self.system_config.get('protected_files', [])
    
    def is_system_directory(self, path: str) -> bool:
        """Check if a path is a protected system directory."""
        system_dirs = self.get_system_directories()
        return any(path.startswith(dir_path) for dir_path in system_dirs)
    
    def is_protected_file(self, filename: str) -> bool:
        """Check if a file is protected."""
        protected_files = self.get_protected_files()
        return filename in protected_files
    
    def _generate_context_hash(self, context: List[Dict[str, Any]]) -> str:
        """Generate hash for context to detect changes."""
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation: 1 token ≈ 4 characters)."""
        if not text:
            return 0
        # Rough estimation: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def _get_context_tokens(self, context_entries: List[Dict[str, Any]]) -> int:
        """Calculate total token count for context entries."""
        total_tokens = 0
        for entry in context_entries:
            content = entry.get('content', '')
            entry_type = entry.get('entry_type', '')
            # Count tokens for content and entry type
            total_tokens += self._count_tokens(content) + self._count_tokens(entry_type)
        return total_tokens
    
    def _summarize_context(self, context: List[Dict[str, Any]]) -> str:
        """Summarize context when it gets too long."""
        if len(context) <= 50:
            return "Context is within limits, no summarization needed."
        
        # Group entries by type
        actions = [entry for entry in context if entry.get('entry_type') == 'action']
        results = [entry for entry in context if entry.get('entry_type') == 'result']
        errors = [entry for entry in context if entry.get('entry_type') == 'error']
        
        summary_parts = []
        
        if actions:
            summary_parts.append(f"Performed {len(actions)} actions")
        
        if results:
            summary_parts.append(f"Completed {len(results)} tasks successfully")
        
        if errors:
            summary_parts.append(f"Encountered {len(errors)} errors")
        
        # Add recent important entries
        recent_entries = context[-10:]
        recent_summary = []
        for entry in recent_entries:
            if entry.get('entry_type') in ['action', 'result']:
                content = entry.get('content', '')[:100]
                recent_summary.append(f"- {entry.get('entry_type')}: {content}")
        
        if recent_summary:
            summary_parts.append("Recent activities:\n" + "\n".join(recent_summary))
        
        return "\n".join(summary_parts)
    
    def create_project(self, project_name: str, project_path: str) -> str:
        """Create a new project context."""
        project_id = f"proj_{int(time.time())}_{hashlib.md5(project_name.encode()).hexdigest()[:8]}"
        
        project = ProjectContext(
            project_id=project_id,
            project_name=project_name,
            project_path=project_path,
            created_at=time.time(),
            last_accessed=time.time(),
            current_directory=project_path,
            full_context=[],  # Complete history from start to finish
            actual_context=[],  # Current working context
            context_summary="",  # Will be populated when actual_context gets summarized
            context_hash="",
            is_active=True
        )
        
        self.projects[project_id] = project
        self.current_project = project
        self._save_projects()
        self._save_context()
        
        self.logger.success(f"Created project '{project_name}' at {project_path}")
        return project_id
    
    def set_active_project(self, project_id: str) -> bool:
        """Set active project by ID."""
        if project_id in self.projects:
            self.current_project = self.projects[project_id]
            self.current_project.last_accessed = time.time()
            self._save_context()
            self.logger.info(f"Activated project: {self.current_project.project_name}")
            return True
        return False
    
    def get_active_project(self) -> Optional[ProjectContext]:
        """Get currently active project."""
        return self.current_project
    
    def get_context_percentage(self) -> float:
        """Get current context percentage based on tokens (0-100)."""
        if not self.current_project:
            return 0.0
        
        # Calculate percentage based on tokens in actual_context
        actual_tokens = self._get_context_tokens(self.current_project.actual_context)
        token_percentage = (actual_tokens / self.max_context_tokens) * 100
        
        # Only return 100% when token limit is actually reached
        return min(100.0, token_percentage)
    
    def add_context_entry(self, entry_type: str, content: str, metadata: Dict[str, Any] = None):
        """Add a new context entry to both full_context and actual_context."""
        if not self.current_project:
            self.logger.warning("No active project, creating default project")
            self.create_project("Default Project", str(settings.BASE_DIR))
        
        entry = ContextEntry(
            timestamp=time.time(),
            entry_type=entry_type,
            content=content,
            metadata=metadata or {},
            entry_id=f"entry_{int(time.time())}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        )
        
        entry_dict = asdict(entry)
        
        # Always add to full_context (never summarized)
        self.current_project.full_context.append(entry_dict)
        
        # Add to actual_context (gets summarized when full)
        self.current_project.actual_context.append(entry_dict)
        
        self.current_project.last_accessed = time.time()
        
        # Calculate and display context percentage based on tokens
        context_percentage = self._get_actual_context_percentage()
        actual_tokens = self._get_context_tokens(self.current_project.actual_context)
        print(f"📊 Context: {context_percentage:.1f}% ({actual_tokens}/{self.max_context_tokens} tokens)")
        
        # Check if actual_context needs summarization (only when token limit is reached)
        if actual_tokens >= self.max_context_tokens:
            print("🔄 Context at 100% - Summarizing actual context...")
            self._summarize_actual_context()
            new_percentage = self._get_actual_context_percentage()
            new_tokens = self._get_context_tokens(self.current_project.actual_context)
            print(f"📊 Context after summarization: {new_percentage:.1f}% ({new_tokens}/{self.max_context_tokens} tokens)")
        
        self._save_context()
        self.logger.debug(f"Added context entry: {entry_type}")
    
    def _get_actual_context_percentage(self) -> float:
        """Get current actual context percentage based on tokens (0-100)."""
        if not self.current_project:
            return 0.0
        
        # Calculate percentage based on tokens in actual_context
        actual_tokens = self._get_context_tokens(self.current_project.actual_context)
        token_percentage = (actual_tokens / self.max_context_tokens) * 100
        
        # Only return 100% when token limit is actually reached
        return min(100.0, token_percentage)
    
    def _summarize_actual_context(self):
        """Summarize actual_context when token limit is reached."""
        if not self.current_project or not self.current_project.actual_context:
            return
        
        # Create summary of actual_context
        summary_entries = []
        for entry in self.current_project.actual_context:
            timestamp = entry.get('timestamp', 0)
            time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S') if timestamp else 'Unknown'
            summary_entries.append(f"[{time_str}] {entry.get('entry_type', 'unknown')}: {entry.get('content', '')[:100]}...")
        
        # Create a comprehensive summary
        summary_content = f"Summarized {len(self.current_project.actual_context)} entries:\n" + "\n".join(summary_entries)
        
        # Update context_summary
        self.current_project.context_summary = summary_content
        
        # Keep only the most recent entries that fit within 20% of token limit
        target_tokens = int(self.max_context_tokens * 0.2)  # 20% of max tokens
        kept_entries = []
        current_tokens = 0
        
        # Keep entries from the end (most recent) until we reach target tokens
        for entry in reversed(self.current_project.actual_context):
            entry_tokens = self._count_tokens(entry.get('content', '')) + self._count_tokens(entry.get('entry_type', ''))
            if current_tokens + entry_tokens <= target_tokens:
                kept_entries.insert(0, entry)  # Insert at beginning to maintain order
                current_tokens += entry_tokens
            else:
                break
        
        self.current_project.actual_context = kept_entries
        
        self.logger.info(f"Summarized actual context: {len(summary_entries)} entries compressed to {len(kept_entries)} entries ({current_tokens} tokens)")
    
    def _summarize_and_compress_context(self):
        """Summarize and compress context when it gets too large."""
        if not self.current_project:
            return
        
        # Get all entries for summarization
        all_entries = self.current_project.full_context.copy()
        
        # Create summary of all context
        summary = self._summarize_context(all_entries)
        
        # Update context summary
        if self.current_project.context_summary:
            self.current_project.context_summary += f"\n\n--- Previous Session Summary ---\n{summary}"
        else:
            self.current_project.context_summary = summary
        
        # Clear all entries and keep only a few recent ones (max 20)
        recent_entries = self.current_project.full_context[-20:] if len(self.current_project.full_context) > 20 else self.current_project.full_context
        self.current_project.full_context = recent_entries
        
        # Update hash
        self.current_project.context_hash = self._generate_context_hash(self.current_project.full_context)
        
        self.logger.info(f"Context summarized and compressed from {len(all_entries)} entries to {len(self.current_project.full_context)} entries")
    
    def get_context_summary(self) -> str:
        """Get current context summary."""
        if not self.current_project:
            return "No active project context."
        
        full_tokens = self._get_context_tokens(self.current_project.full_context)
        actual_tokens = self._get_context_tokens(self.current_project.actual_context)
        
        summary_parts = [
            f"Project: {self.current_project.project_name}",
            f"Path: {self.current_project.project_path}",
            f"Current Directory: {self.current_project.current_directory}",
            f"Last Accessed: {datetime.fromtimestamp(self.current_project.last_accessed).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Full Context: {len(self.current_project.full_context)} entries, {full_tokens} tokens (complete history)",
            f"Actual Context: {len(self.current_project.actual_context)} entries, {actual_tokens}/{self.max_context_tokens} tokens (current working context)"
        ]
        
        if self.current_project.context_summary:
            summary_parts.append(f"\nContext Summary:\n{self.current_project.context_summary}")
        
        return "\n".join(summary_parts)
    
    def get_recent_context(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent context entries from actual_context."""
        if not self.current_project:
            return []
        
        return self.current_project.actual_context[-count:]
    
    def search_context(self, query: str) -> List[Dict[str, Any]]:
        """Search through context entries."""
        if not self.current_project:
            return []
        
        results = []
        query_lower = query.lower()
        
        for entry in self.current_project.full_context:
            if (query_lower in entry.get('content', '').lower() or 
                query_lower in entry.get('entry_type', '').lower()):
                results.append(entry)
        
        return results
    
    def search_context_by_time(self, start_time: float = None, end_time: float = None, entry_type: str = None) -> List[Dict[str, Any]]:
        """Search context entries by time range and optional entry type."""
        if not self.current_project:
            return []
        
        results = []
        current_time = time.time()
        
        # Default to last 24 hours if no time specified
        if start_time is None:
            start_time = current_time - (24 * 60 * 60)  # 24 hours ago
        if end_time is None:
            end_time = current_time
        
        for entry in self.current_project.full_context:
            entry_timestamp = entry.get('timestamp', 0)
            entry_type_filter = entry.get('entry_type', '')
            
            # Check time range
            if start_time <= entry_timestamp <= end_time:
                # Check entry type filter if specified
                if entry_type is None or entry_type_filter == entry_type:
                    results.append(entry)
        
        return results
    
    def get_context_by_date(self, date_str: str) -> List[Dict[str, Any]]:
        """Get all context entries for a specific date (YYYY-MM-DD format)."""
        if not self.current_project:
            return []
        
        try:
            from datetime import datetime
            target_date = datetime.strptime(date_str, '%Y-%m-%d')
            start_time = target_date.timestamp()
            end_time = start_time + (24 * 60 * 60)  # Next day
            
            return self.search_context_by_time(start_time, end_time)
        except ValueError:
            self.logger.error(f"Invalid date format: {date_str}. Use YYYY-MM-DD format.")
            return []
    
    def get_context_by_hour(self, hour: int) -> List[Dict[str, Any]]:
        """Get all context entries for a specific hour (0-23)."""
        if not self.current_project:
            return []
        
        current_time = time.time()
        today = datetime.fromtimestamp(current_time).replace(hour=hour, minute=0, second=0, microsecond=0)
        start_time = today.timestamp()
        end_time = start_time + (60 * 60)  # Next hour
        
        return self.search_context_by_time(start_time, end_time)
    
    def update_current_directory(self, new_directory: str):
        """Update current working directory."""
        if self.current_project:
            self.current_project.current_directory = new_directory
            self.add_context_entry('info', f"Changed directory to: {new_directory}")
            self._save_context()
    
    def get_project_list(self) -> List[Dict[str, Any]]:
        """Get list of all projects."""
        return [
            {
                'id': project.project_id,
                'name': project.project_name,
                'path': project.project_path,
                'last_accessed': project.last_accessed,
                'is_active': project.project_id == (self.current_project.project_id if self.current_project else None)
            }
            for project in self.projects.values()
        ]
    
    def cleanup_old_projects(self, days_old: int = 30):
        """Clean up old inactive projects."""
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        
        to_remove = []
        for project_id, project in self.projects.items():
            if (not project.is_active and 
                project.last_accessed < cutoff_time and 
                project_id != (self.current_project.project_id if self.current_project else None)):
                to_remove.append(project_id)
        
        for project_id in to_remove:
            del self.projects[project_id]
        
        if to_remove:
            self._save_projects()
            self.logger.info(f"Cleaned up {len(to_remove)} old projects")
    
    def export_context(self, project_id: str = None) -> Dict[str, Any]:
        """Export context for backup or analysis."""
        if project_id and project_id in self.projects:
            project = self.projects[project_id]
        elif self.current_project:
            project = self.current_project
        else:
            return {}
        
        return {
            'project': asdict(project),
            'export_timestamp': time.time(),
            'export_version': '1.0'
        }
    
    def import_context(self, context_data: Dict[str, Any]) -> bool:
        """Import context from backup."""
        try:
            if 'project' in context_data:
                project_data = context_data['project']
                project = ProjectContext(**project_data)
                self.projects[project.project_id] = project
                self._save_projects()
                self.logger.success(f"Imported project: {project.project_name}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to import context: {e}")
        return False
