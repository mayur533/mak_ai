"""
Advanced Production-Ready Context Management System for AI Assistant.
Comprehensive context management with enterprise features, analytics, and optimization.
"""

import json
import os
import time
import gzip
import threading
import asyncio
import hashlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import pickle
import zlib
from collections import defaultdict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config.settings import settings
from src.logging.logger import logger


class ContextType(Enum):
    """Types of context entries."""
    ACTION = "action"
    RESULT = "result"
    ERROR = "error"
    INFO = "info"
    WARNING = "warning"
    DEBUG = "debug"
    SYSTEM = "system"
    USER_INPUT = "user_input"
    AI_RESPONSE = "ai_response"
    TOOL_EXECUTION = "tool_execution"
    FILE_OPERATION = "file_operation"
    NETWORK_REQUEST = "network_request"
    DATABASE_OPERATION = "database_operation"


class Priority(Enum):
    """Priority levels for context entries."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ContextEntry:
    """Advanced context entry with metadata and analytics."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    context_type: ContextType = ContextType.INFO
    priority: Priority = Priority.NORMAL
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    source: str = "system"
    duration: Optional[float] = None
    success: Optional[bool] = None
    tokens_used: int = 0
    memory_impact: int = 0
    related_entries: List[str] = field(default_factory=list)
    compressed: bool = False
    version: int = 1


@dataclass
class SessionContext:
    """Advanced session context with analytics and optimization."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    current_directory: str = ""
    context_entries: List[ContextEntry] = field(default_factory=list)
    context_summary: str = ""
    context_hash: str = ""
    is_active: bool = True
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    memory_usage: int = 0
    token_usage: int = 0
    compression_ratio: float = 0.0
    last_optimization: float = 0.0


@dataclass
class ContextAnalytics:
    """Analytics data for context management."""
    total_entries: int = 0
    entries_by_type: Dict[str, int] = field(default_factory=dict)
    entries_by_priority: Dict[str, int] = field(default_factory=dict)
    average_entry_size: float = 0.0
    compression_savings: float = 0.0
    memory_efficiency: float = 0.0
    token_efficiency: float = 0.0
    error_rate: float = 0.0
    success_rate: float = 0.0
    performance_trends: List[Dict[str, Any]] = field(default_factory=list)


class AdvancedContextManager:
    """Production-ready context manager with enterprise features."""

    def __init__(self):
        """Initialize the advanced context manager."""
        self.logger = logger
        self.db_path = settings.BASE_DIR / "db" / "context.db"
        self.cache_path = settings.BASE_DIR / "db" / "context_cache.pkl"
        self.config_path = settings.BASE_DIR / "db" / "context_config.json"
        
        # Configuration
        self.max_context_tokens = 50000
        self.max_context_entries = 10000
        self.max_memory_usage = 100 * 1024 * 1024  # 100MB
        self.compression_threshold = 1024
        self.cache_size = 1000
        self.optimization_interval = 3600  # 1 hour
        
        # State management
        self.current_session: Optional[SessionContext] = None
        self.sessions: Dict[str, SessionContext] = {}
        self.analytics = ContextAnalytics()
        
        # Performance optimization
        self._context_cache = {}
        self._cache_lock = threading.RLock()
        self._db_lock = threading.RLock()
        self._optimization_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Compression settings
        self.compression_enabled = True
        self.compression_level = 6
        self.adaptive_compression = True
        
        # Initialize system
        self._ensure_directories()
        self._init_database()
        self._load_configuration()
        self._load_sessions()
        self._load_analytics()
        
        # Start background tasks
        self._start_background_tasks()

    def _ensure_directories(self):
        """Ensure all required directories exist."""
        self.db_path.parent.mkdir(exist_ok=True)
        self.cache_path.parent.mkdir(exist_ok=True)
        self.config_path.parent.mkdir(exist_ok=True)

    def _init_database(self):
        """Initialize SQLite database for context storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS context_entries (
                        entry_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        timestamp REAL,
                        context_type TEXT,
                        priority INTEGER,
                        content TEXT,
                        metadata TEXT,
                        tags TEXT,
                        source TEXT,
                        duration REAL,
                        success INTEGER,
                        tokens_used INTEGER,
                        memory_impact INTEGER,
                        related_entries TEXT,
                        compressed INTEGER,
                        version INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at REAL,
                        last_accessed REAL,
                        current_directory TEXT,
                        context_summary TEXT,
                        context_hash TEXT,
                        is_active INTEGER,
                        session_metadata TEXT,
                        performance_metrics TEXT,
                        memory_usage INTEGER,
                        token_usage INTEGER,
                        compression_ratio REAL,
                        last_optimization REAL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS analytics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        metric_name TEXT,
                        metric_value REAL,
                        metadata TEXT
                    )
                """)
                
                # Create indexes for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_context_session ON context_entries(session_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_context_timestamp ON context_entries(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_context_type ON context_entries(context_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics(timestamp)")
                
                conn.commit()
                self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")

    def _load_configuration(self):
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.max_context_tokens = config.get('max_context_tokens', self.max_context_tokens)
                    self.max_context_entries = config.get('max_context_entries', self.max_context_entries)
                    self.max_memory_usage = config.get('max_memory_usage', self.max_memory_usage)
                    self.compression_enabled = config.get('compression_enabled', self.compression_enabled)
                    self.compression_level = config.get('compression_level', self.compression_level)
                    self.adaptive_compression = config.get('adaptive_compression', self.adaptive_compression)
                    self.cache_size = config.get('cache_size', self.cache_size)
                    self.optimization_interval = config.get('optimization_interval', self.optimization_interval)
        except Exception as e:
            self.logger.warning(f"Failed to load configuration: {e}")

    def _save_configuration(self):
        """Save configuration to file."""
        try:
            config = {
                'max_context_tokens': self.max_context_tokens,
                'max_context_entries': self.max_context_entries,
                'max_memory_usage': self.max_memory_usage,
                'compression_enabled': self.compression_enabled,
                'compression_level': self.compression_level,
                'adaptive_compression': self.adaptive_compression,
                'cache_size': self.cache_size,
                'optimization_interval': self.optimization_interval
            }
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save configuration: {e}")

    def _load_sessions(self):
        """Load sessions from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM sessions")
                for row in cursor.fetchall():
                    session_data = dict(row)
                    session_data['context_entries'] = []
                    session_data['session_metadata'] = json.loads(session_data.get('session_metadata', '{}'))
                    session_data['performance_metrics'] = json.loads(session_data.get('performance_metrics', '{}'))
                    session = SessionContext(**session_data)
                    self.sessions[session.session_id] = session
                    
                    if session.is_active:
                        self.current_session = session
                        
            self.logger.info(f"Loaded {len(self.sessions)} sessions")
        except Exception as e:
            self.logger.error(f"Failed to load sessions: {e}")

    def _load_analytics(self):
        """Load analytics data from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT metric_name, metric_value, timestamp 
                    FROM analytics 
                    ORDER BY timestamp DESC 
                    LIMIT 1000
                """)
                
                metrics = defaultdict(list)
                for row in cursor.fetchall():
                    metrics[row['metric_name']].append({
                        'value': row['metric_value'],
                        'timestamp': row['timestamp']
                    })
                
                # Calculate analytics
                self.analytics.total_entries = len(metrics.get('total_entries', []))
                self.analytics.compression_savings = np.mean([m['value'] for m in metrics.get('compression_savings', [])]) if metrics.get('compression_savings') else 0.0
                self.analytics.memory_efficiency = np.mean([m['value'] for m in metrics.get('memory_efficiency', [])]) if metrics.get('memory_efficiency') else 0.0
                self.analytics.token_efficiency = np.mean([m['value'] for m in metrics.get('token_efficiency', [])]) if metrics.get('token_efficiency') else 0.0
                
        except Exception as e:
            self.logger.warning(f"Failed to load analytics: {e}")

    def _start_background_tasks(self):
        """Start background optimization and maintenance tasks."""
        def background_optimizer():
            while True:
                try:
                    time.sleep(self.optimization_interval)
                    self._optimize_context()
                    self._cleanup_old_data()
                    self._update_analytics()
                except Exception as e:
                    self.logger.error(f"Background optimizer error: {e}")
        
        # Start background thread
        threading.Thread(target=background_optimizer, daemon=True).start()

    def create_session(self, session_name: str = "default", metadata: Dict[str, Any] = None) -> str:
        """Create a new session with advanced features."""
        session_id = str(uuid.uuid4())
        session = SessionContext(
            session_id=session_id,
            current_directory=str(settings.BASE_DIR),
            session_metadata=metadata or {},
            performance_metrics={}
        )
        
        self.sessions[session_id] = session
        self.current_session = session
        
        # Save to database
        self._save_session_to_db(session)
        
        self.logger.info(f"Created new session: {session_id}")
        return session_id

    def set_active_session(self, session_id: str) -> bool:
        """Set the active session."""
        try:
            if session_id in self.sessions:
                self.current_session = self.sessions[session_id]
                self.current_session.last_accessed = time.time()
                self._save_session_to_db(self.current_session)
                self.logger.info(f"Set active session: {session_id}")
                return True
            else:
                self.logger.warning(f"Session not found: {session_id}")
                return False
        except Exception as e:
            self.logger.error(f"Error setting active session: {e}")
            return False

    def get_active_session(self) -> Optional[SessionContext]:
        """Get the currently active session."""
        return self.current_session

    def get_session_list(self) -> List[Dict[str, Any]]:
        """Get list of all sessions."""
        try:
            sessions = []
            for session_id, session in self.sessions.items():
                sessions.append({
                    "session_id": session_id,
                    "session_name": getattr(session, 'session_name', 'Unknown'),
                    "created_at": getattr(session, 'created_at', time.time()),
                    "last_accessed": getattr(session, 'last_accessed', time.time()),
                    "entry_count": len(session.context_entries),
                    "metadata": getattr(session, 'session_metadata', {})
                })
            return sorted(sessions, key=lambda x: x["last_accessed"], reverse=True)
        except Exception as e:
            self.logger.error(f"Error getting session list: {e}")
            return []

    def add_context_entry(
        self,
        context_type: ContextType,
        content: str,
        priority: Priority = Priority.NORMAL,
        metadata: Dict[str, Any] = None,
        tags: Set[str] = None,
        source: str = "system",
        duration: float = None,
        success: bool = None,
        related_entries: List[str] = None
    ) -> str:
        """Add a context entry with advanced features."""
        if not self.current_session:
            self.create_session()
        
        # Calculate tokens and memory impact
        tokens_used = self._estimate_tokens(content)
        memory_impact = len(content.encode('utf-8'))
        
        # Create entry
        entry = ContextEntry(
            context_type=context_type,
            priority=priority,
            content=content,
            metadata=metadata or {},
            tags=tags or set(),
            source=source,
            duration=duration,
            success=success,
            tokens_used=tokens_used,
            memory_impact=memory_impact,
            related_entries=related_entries or []
        )
        
        # Add to session
        self.current_session.context_entries.append(entry)
        self.current_session.last_accessed = time.time()
        self.current_session.memory_usage += memory_impact
        self.current_session.token_usage += tokens_used
        
        # Save to database
        self._save_entry_to_db(entry)
        
        # Update analytics
        self._update_entry_analytics(entry)
        
        # Check if optimization is needed
        if self._should_optimize():
            self._optimize_context()
        
        self.logger.debug(f"Added context entry: {context_type.value}")
        return entry.entry_id

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if not text:
            return 0
        # More accurate estimation: 1 token ≈ 4 characters for English
        return max(1, len(text) // 4)

    def _should_optimize(self) -> bool:
        """Check if context optimization is needed."""
        if not self.current_session:
            return False
        
        return (
            self.current_session.token_usage >= self.max_context_tokens or
            self.current_session.memory_usage >= self.max_memory_usage or
            len(self.current_session.context_entries) >= self.max_context_entries or
            time.time() - self.current_session.last_optimization >= self.optimization_interval
        )

    def _optimize_context(self):
        """Optimize context for performance and memory usage."""
        if not self.current_session:
            return

        with self._optimization_lock:
            try:
                # Compress large entries
                compressed_count = 0
                for entry in self.current_session.context_entries:
                    if not entry.compressed and len(entry.content) > self.compression_threshold:
                        entry.content = self._compress_text(entry.content)
                        entry.compressed = True
                        compressed_count += 1
                
                # Summarize old entries
                if len(self.current_session.context_entries) > self.max_context_entries * 0.8:
                    self._summarize_old_entries()
                
                # Update compression ratio
                original_size = sum(entry.memory_impact for entry in self.current_session.context_entries)
                # Convert entries to serializable format
                serializable_entries = []
                for entry in self.current_session.context_entries:
                    entry_dict = asdict(entry)
                    entry_dict['context_type'] = entry.context_type.value
                    entry_dict['priority'] = entry.priority.value
                    entry_dict['tags'] = list(entry.tags)
                    serializable_entries.append(entry_dict)
                compressed_size = len(json.dumps(serializable_entries).encode())
                self.current_session.compression_ratio = 1 - (compressed_size / original_size) if original_size > 0 else 0
                
                # Update last optimization time
                self.current_session.last_optimization = time.time()
                
                # Save optimized session
                self._save_session_to_db(self.current_session)
                
                self.logger.info(f"Context optimized: {compressed_count} entries compressed")
                
            except Exception as e:
                self.logger.error(f"Context optimization failed: {e}")

    def _compress_text(self, text: str) -> str:
        """Compress text using zlib."""
        try:
            compressed = zlib.compress(text.encode('utf-8'), level=self.compression_level)
            return f"COMPRESSED:{compressed.hex()}"
        except Exception as e:
            self.logger.warning(f"Text compression failed: {e}")
            return text

    def _decompress_text(self, text: str) -> str:
        """Decompress text."""
        try:
            if text.startswith("COMPRESSED:"):
                compressed_hex = text[11:]
                compressed = bytes.fromhex(compressed_hex)
                return zlib.decompress(compressed).decode('utf-8')
            return text
        except Exception as e:
            self.logger.warning(f"Text decompression failed: {e}")
            return text

    def _summarize_old_entries(self):
        """Summarize old context entries to save space."""
        if not self.current_session:
            return

        # Keep recent entries and summarize older ones
        recent_count = self.max_context_entries // 2
        recent_entries = self.current_session.context_entries[-recent_count:]
        old_entries = self.current_session.context_entries[:-recent_count]
        
        if old_entries:
            # Create summary of old entries
            summary_content = self._create_entries_summary(old_entries)
            
            # Create summary entry
            summary_entry = ContextEntry(
                context_type=ContextType.SYSTEM,
                priority=Priority.LOW,
                content=f"Summarized {len(old_entries)} entries: {summary_content}",
                metadata={'summarized_count': len(old_entries)},
                tags={'summary'},
                source='optimizer'
            )
            
            # Replace old entries with summary
            self.current_session.context_entries = [summary_entry] + recent_entries
            
            self.logger.info(f"Summarized {len(old_entries)} old entries")

    def _create_entries_summary(self, entries: List[ContextEntry]) -> str:
        """Create a summary of multiple entries."""
        type_counts = defaultdict(int)
        priority_counts = defaultdict(int)
        
        for entry in entries:
            type_counts[entry.context_type.value] += 1
            priority_counts[entry.priority.value] += 1
        
        summary_parts = []
        for context_type, count in type_counts.items():
            summary_parts.append(f"{count} {context_type}")
        
        return f"Activities: {', '.join(summary_parts)}"

    def _save_entry_to_db(self, entry: ContextEntry):
        """Save context entry to database."""
        try:
            # Handle enum conversion safely
            context_type_value = entry.context_type.value if hasattr(entry.context_type, 'value') else str(entry.context_type)
            priority_value = entry.priority.value if hasattr(entry.priority, 'value') else str(entry.priority)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO context_entries 
                    (entry_id, session_id, timestamp, context_type, priority, content, 
                     metadata, tags, source, duration, success, tokens_used, 
                     memory_impact, related_entries, compressed, version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.entry_id,
                    self.current_session.session_id,
                    entry.timestamp,
                    context_type_value,
                    priority_value,
                    entry.content,
                    json.dumps(entry.metadata),
                    json.dumps(list(entry.tags)),
                    entry.source,
                    entry.duration,
                    entry.success,
                    entry.tokens_used,
                    entry.memory_impact,
                    json.dumps(entry.related_entries),
                    entry.compressed,
                    entry.version
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save entry to database: {e}")

    def _save_session_to_db(self, session: SessionContext):
        """Save session to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO sessions 
                    (session_id, created_at, last_accessed, current_directory, 
                     context_summary, context_hash, is_active, session_metadata, 
                     performance_metrics, memory_usage, token_usage, 
                     compression_ratio, last_optimization)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.created_at,
                    session.last_accessed,
                    session.current_directory,
                    session.context_summary,
                    session.context_hash,
                    session.is_active,
                    json.dumps(session.session_metadata),
                    json.dumps(session.performance_metrics),
                    session.memory_usage,
                    session.token_usage,
                    session.compression_ratio,
                    session.last_optimization
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save session to database: {e}")

    def _update_entry_analytics(self, entry: ContextEntry):
        """Update analytics with new entry data."""
        self.analytics.total_entries += 1
        self.analytics.entries_by_type[entry.context_type.value] = self.analytics.entries_by_type.get(entry.context_type.value, 0) + 1
        self.analytics.entries_by_priority[entry.priority.value] = self.analytics.entries_by_priority.get(entry.priority.value, 0) + 1
        
        # Update success/error rates
        if entry.success is not None:
            if entry.success:
                self.analytics.success_rate = (self.analytics.success_rate * (self.analytics.total_entries - 1) + 1) / self.analytics.total_entries
        else:
                self.analytics.error_rate = (self.analytics.error_rate * (self.analytics.total_entries - 1) + 1) / self.analytics.total_entries

    def _update_analytics(self):
        """Update analytics metrics in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                current_time = time.time()
                
                # Save various metrics
                metrics = [
                    ('total_entries', self.analytics.total_entries, current_time),
                    ('compression_savings', self.analytics.compression_savings, current_time),
                    ('memory_efficiency', self.analytics.memory_efficiency, current_time),
                    ('token_efficiency', self.analytics.token_efficiency, current_time),
                    ('error_rate', self.analytics.error_rate, current_time),
                    ('success_rate', self.analytics.success_rate, current_time)
                ]
                
                conn.executemany("""
                    INSERT INTO analytics (metric_name, metric_value, timestamp)
                    VALUES (?, ?, ?)
                """, metrics)
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to update analytics: {e}")

    def _cleanup_old_data(self):
        """Clean up old data to maintain performance."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clean up old analytics data (keep last 30 days)
                cutoff_time = time.time() - (30 * 24 * 60 * 60)
                conn.execute("DELETE FROM analytics WHERE timestamp < ?", (cutoff_time,))
                
                # Clean up old context entries (keep last 7 days)
                cutoff_time = time.time() - (7 * 24 * 60 * 60)
                conn.execute("DELETE FROM context_entries WHERE timestamp < ?", (cutoff_time,))
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")

    def get_context_summary(self) -> str:
        """Get comprehensive context summary."""
        if not self.current_session:
            return "No active session."

        summary_parts = [
            f"Session: {self.current_session.session_id[:8]}...",
            f"Created: {datetime.fromtimestamp(self.current_session.created_at).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Last Accessed: {datetime.fromtimestamp(self.current_session.last_accessed).strftime('%Y-%m-%d %H:%M:%S')}",
            f"Current Directory: {self.current_session.current_directory}",
            f"Entries: {len(self.current_session.context_entries)}",
            f"Memory Usage: {self.current_session.memory_usage / 1024 / 1024:.2f} MB",
            f"Token Usage: {self.current_session.token_usage}/{self.max_context_tokens}",
            f"Compression Ratio: {self.current_session.compression_ratio:.2%}",
        ]
        
        if self.current_session.context_summary:
            summary_parts.append(f"\nSummary:\n{self.current_session.context_summary}")

        return "\n".join(summary_parts)

    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics data."""
        return {
            'total_entries': self.analytics.total_entries,
            'entries_by_type': dict(self.analytics.entries_by_type),
            'entries_by_priority': dict(self.analytics.entries_by_priority),
            'compression_savings': self.analytics.compression_savings,
            'memory_efficiency': self.analytics.memory_efficiency,
            'token_efficiency': self.analytics.token_efficiency,
            'error_rate': self.analytics.error_rate,
            'success_rate': self.analytics.success_rate,
            'current_session': {
                'memory_usage': self.current_session.memory_usage if self.current_session else 0,
                'token_usage': self.current_session.token_usage if self.current_session else 0,
                'compression_ratio': self.current_session.compression_ratio if self.current_session else 0.0
            }
        }

    def search_context(
        self,
        query: str,
        context_type: Optional[ContextType] = None,
        priority: Optional[Priority] = None,
        tags: Optional[Set[str]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100
    ) -> List[ContextEntry]:
        """Advanced context search with multiple filters."""
        if not self.current_session:
            return []

        results = []
        query_lower = query.lower()

        for entry in self.current_session.context_entries:
            # Text search
            if query_lower not in entry.content.lower():
                continue
            
            # Type filter
            if context_type and entry.context_type != context_type:
                continue
            
            # Priority filter
            if priority and entry.priority != priority:
                continue
            
            # Tags filter
            if tags and not tags.intersection(entry.tags):
                continue
            
            # Time filter
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            
            results.append(entry)
            
            if len(results) >= limit:
                break

        return results

    def get_recent_context(self, count: int = 10) -> List[ContextEntry]:
        """Get recent context entries."""
        if not self.current_session:
            return []
        return self.current_session.context_entries[-count:]

    def clear_context(self):
        """Clear current session context."""
        if self.current_session:
            self.current_session.context_entries.clear()
            self.current_session.memory_usage = 0
            self.current_session.token_usage = 0
            self.current_session.context_summary = ""
            self._save_session_to_db(self.current_session)
            self.logger.info("Context cleared")

    def cleanup_old_sessions(self, max_age_days: int = 30):
        """Clean up old sessions to prevent database bloat."""
        try:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            sessions_to_remove = []
            
            for session_id, session in self.sessions.items():
                if current_time - getattr(session, 'last_accessed', current_time) > max_age_seconds:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.sessions[session_id]
                self.logger.info(f"Cleaned up old session: {session_id}")
            
            if sessions_to_remove:
                self._save_sessions()
                self.logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old sessions: {e}")

    def close(self):
        """Close the context manager and cleanup resources."""
        try:
            # Save all sessions
            for session in self.sessions.values():
                self._save_session_to_db(session)
            
            # Save configuration
            self._save_configuration()
            
            # Close executor
            self._executor.shutdown(wait=True)
            
            self.logger.info("Context manager closed")
        except Exception as e:
            self.logger.error(f"Error closing context manager: {e}")


# Backward compatibility
ContextManager = AdvancedContextManager