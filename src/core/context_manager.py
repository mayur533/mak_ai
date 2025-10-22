"""
Advanced Context Management System - Production Ready
Based on Anthropic's Context Engineering Best Practices and Industry Standards

Key Features:
- Vector-based context storage with embeddings
- Context compaction and summarization
- Structured note-taking and memory tools
- Hybrid retrieval (pre-computed + runtime exploration)
- Minimal high-signal token optimization
- Multi-agent context isolation
- Real-time context relevance scoring
"""

import json
import os
import time
import sqlite3
import asyncio
import threading
import hashlib
import pickle
import zlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
import numpy as np
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import re

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config.settings import settings
from src.logging.logger import logger

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not available, using fallback embeddings")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss not available, using fallback vector search")


class ContextType(Enum):
    """Types of context entries with semantic meaning."""
    USER_INPUT = "user_input"
    AI_RESPONSE = "ai_response"
    TOOL_EXECUTION = "tool_execution"
    SYSTEM_EVENT = "system_event"
    ERROR = "error"
    MEMORY_NOTE = "memory_note"
    COMPACTION = "compaction"
    SEARCH_RESULT = "search_result"
    FILE_OPERATION = "file_operation"
    NETWORK_REQUEST = "network_request"
    DATABASE_OPERATION = "database_operation"
    AGENT_COORDINATION = "agent_coordination"


class Priority(Enum):
    """Priority levels for context entries."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    ARCHIVE = 5


class ContextRelevance(Enum):
    """Relevance levels for context retrieval."""
    ESSENTIAL = 1
    IMPORTANT = 2
    RELEVANT = 3
    CONTEXTUAL = 4
    BACKGROUND = 5


@dataclass
class ContextEntry:
    """Enhanced context entry with vector embeddings and relevance scoring."""
    entry_id: str
    context_type: ContextType
    content: str
    priority: Priority = Priority.NORMAL
    relevance: ContextRelevance = ContextRelevance.RELEVANT
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    source: str = "system"
    duration: float = 0.0
    success: bool = True
    tokens_used: int = 0
    memory_impact: int = 0
    related_entries: List[str] = field(default_factory=list)
    compressed: bool = False
    version: int = 1
    embedding: Optional[np.ndarray] = None
    semantic_hash: str = ""
    context_window_position: int = 0
    attention_weight: float = 1.0
    decay_factor: float = 1.0
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class SessionContext:
    """Enhanced session context with advanced memory management."""
    session_id: str
    session_name: str = "default"
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
    compaction_count: int = 0
    last_compaction: float = 0.0
    context_window_size: int = 50000
    current_window_usage: int = 0
    memory_notes: Dict[str, str] = field(default_factory=dict)
    agent_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextAnalytics:
    """Advanced analytics for context management."""
    total_entries: int = 0
    entries_by_type: Dict[str, int] = field(default_factory=dict)
    entries_by_priority: Dict[str, int] = field(default_factory=dict)
    entries_by_relevance: Dict[str, int] = field(default_factory=dict)
    average_entry_size: float = 0.0
    compression_savings: float = 0.0
    memory_efficiency: float = 0.0
    token_efficiency: float = 0.0
    context_retrieval_accuracy: float = 0.0
    compaction_effectiveness: float = 0.0
    attention_distribution: Dict[str, float] = field(default_factory=dict)
    performance_trends: List[Dict[str, Any]] = field(default_factory=list)
    error_rate: float = 0.0
    success_rate: float = 0.0


class AdvancedContextManager:
    """
    Production-ready context manager implementing Anthropic's best practices:
    - Vector-based context storage with embeddings
    - Context compaction and summarization
    - Structured note-taking and memory tools
    - Hybrid retrieval (pre-computed + runtime exploration)
    - Minimal high-signal token optimization
    """

    def __init__(self):
        """Initialize the advanced context manager."""
        self.logger = logger
        self.db_path = settings.BASE_DIR / "db" / "context.db"
        self.vector_index_path = settings.BASE_DIR / "db" / "context_vectors.index"
        self.memory_notes_path = settings.BASE_DIR / "db" / "memory_notes.json"
        
        # Configuration based on Anthropic's recommendations
        self.max_context_tokens = 50000  # Optimal context window
        self.max_context_entries = 10000
        self.max_memory_usage = 200 * 1024 * 1024  # 200MB
        self.compaction_threshold = 0.8  # Compact when 80% full
        self.relevance_threshold = 0.7  # Minimum relevance for retrieval
        self.attention_decay_rate = 0.95  # How quickly attention weights decay
        self.compaction_ratio = 0.3  # Keep 30% of original context after compaction
        
        # State management
        self.current_session: Optional[SessionContext] = None
        self.sessions: Dict[str, SessionContext] = {}
        self.analytics = ContextAnalytics()
        
        # Vector storage
        self.embedding_model = None
        self.vector_index = None
        self.entry_embeddings: Dict[str, np.ndarray] = {}
        
        # Performance optimization
        self._context_cache = {}
        self._cache_lock = threading.RLock()
        self._db_lock = threading.RLock()
        self._vector_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Memory notes system
        self.memory_notes: Dict[str, str] = {}
        
        # Initialize system
        self._ensure_directories()
        self._init_database()
        self._init_embedding_model()
        self._init_vector_index()
        self._load_sessions()
        self._load_memory_notes()
        self._load_analytics()
        
        # Start background optimization
        self._start_background_tasks()

    def _ensure_directories(self):
        """Ensure all required directories exist."""
        self.db_path.parent.mkdir(exist_ok=True)
        self.vector_index_path.parent.mkdir(exist_ok=True)
        self.memory_notes_path.parent.mkdir(exist_ok=True)

    def _init_database(self):
        """Initialize SQLite database with optimized schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Context entries table with vector support
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS context_entries (
                        entry_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        timestamp REAL,
                        context_type TEXT,
                        priority INTEGER,
                        relevance INTEGER,
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
                        embedding BLOB,
                        semantic_hash TEXT,
                        context_window_position INTEGER,
                        attention_weight REAL,
                        decay_factor REAL,
                        last_accessed REAL,
                        access_count INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes separately
                conn.execute("CREATE INDEX IF NOT EXISTS idx_session_timestamp ON context_entries (session_id, timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_context_type ON context_entries (context_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_priority ON context_entries (priority)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_relevance ON context_entries (relevance)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_semantic_hash ON context_entries (semantic_hash)")
                
                # Sessions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id TEXT PRIMARY KEY,
                        session_name TEXT,
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
                        compaction_count INTEGER,
                        last_compaction REAL,
                        context_window_size INTEGER,
                        current_window_usage INTEGER,
                        created_at_db DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Memory notes table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_notes (
                        note_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        title TEXT,
                        content TEXT,
                        tags TEXT,
                        importance REAL,
                        created_at REAL,
                        last_accessed REAL,
                        access_count INTEGER,
                        created_at_db DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                self.logger.info("Database initialized with advanced schema")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")

    def _init_embedding_model(self):
        """Initialize embedding model for vector operations."""
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Embedding model initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize embedding model: {e}")
                self.embedding_model = None
        else:
            self.logger.warning("Using fallback embedding system")

    def _init_vector_index(self):
        """Initialize FAISS vector index for similarity search."""
        if FAISS_AVAILABLE and self.embedding_model:
            try:
                if self.vector_index_path.exists():
                    self.vector_index = faiss.read_index(str(self.vector_index_path))
                else:
                    # Create new index with 384 dimensions (all-MiniLM-L6-v2)
                    self.vector_index = faiss.IndexFlatIP(384)
                self.logger.info("Vector index initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize vector index: {e}")
                self.vector_index = None

    def _load_sessions(self):
        """Load sessions from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM sessions")
                for row in cursor.fetchall():
                    session = SessionContext(
                        session_id=row[0],
                        session_name=row[1] or "default",
                        created_at=row[2],
                        last_accessed=row[3],
                        current_directory=row[4] or "",
                        context_summary=row[5] or "",
                        context_hash=row[6] or "",
                        is_active=bool(row[7]),
                        session_metadata=json.loads(row[8]) if row[8] else {},
                        performance_metrics=json.loads(row[9]) if row[9] else {},
                        memory_usage=row[10] or 0,
                        token_usage=row[11] or 0,
                        compaction_count=row[12] or 0,
                        last_compaction=row[13] or 0,
                        context_window_size=row[14] or 50000,
                        current_window_usage=row[15] or 0
                    )
                    self.sessions[session.session_id] = session
                    
                    if session.is_active:
                        self.current_session = session
                        
            self.logger.info(f"Loaded {len(self.sessions)} sessions")
        except Exception as e:
            self.logger.error(f"Failed to load sessions: {e}")

    def _load_memory_notes(self):
        """Load memory notes from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM memory_notes")
                for row in cursor.fetchall():
                    note_id = row[0]
                    self.memory_notes[note_id] = {
                        "title": row[2],
                        "content": row[3],
                        "tags": json.loads(row[4]) if row[4] else [],
                        "importance": row[5],
                        "created_at": row[6],
                        "last_accessed": row[7],
                        "access_count": row[8]
                    }
            self.logger.info(f"Loaded {len(self.memory_notes)} memory notes")
        except Exception as e:
            self.logger.error(f"Failed to load memory notes: {e}")

    def _load_analytics(self):
        """Load analytics from database."""
        try:
            # Load analytics from context entries
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_entries,
                        AVG(LENGTH(content)) as avg_size,
                        AVG(tokens_used) as avg_tokens,
                        AVG(attention_weight) as avg_attention,
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_entries
                    FROM context_entries
                """)
                row = cursor.fetchone()
                if row:
                    self.analytics.total_entries = row[0]
                    self.analytics.average_entry_size = row[1] or 0
                    self.analytics.token_efficiency = row[2] or 0
                    self.analytics.attention_distribution["average"] = row[3] or 0
                    self.analytics.success_rate = (row[4] or 0) / max(row[0], 1)
        except Exception as e:
            self.logger.error(f"Failed to load analytics: {e}")

    def _start_background_tasks(self):
        """Start background optimization tasks."""
        def background_optimizer():
            while True:
                try:
                    time.sleep(3600)  # Run every hour
                    self._optimize_context()
                    self._cleanup_old_entries()
                    self._update_attention_weights()
                except Exception as e:
                    self.logger.error(f"Background optimization error: {e}")
        
        thread = threading.Thread(target=background_optimizer, daemon=True)
        thread.start()

    def create_session(self, session_name: str = "default") -> str:
        """Create a new session with advanced features."""
        session_id = str(uuid.uuid4())
        session = SessionContext(
            session_id=session_id,
            session_name=session_name,
            created_at=time.time(),
            last_accessed=time.time(),
            context_window_size=self.max_context_tokens
        )
        
        self.sessions[session_id] = session
        self.current_session = session
        self._save_session_to_db(session)
        
        self.logger.info(f"Created new session: {session_id}")
        return session_id

    def add_context_entry(
        self,
        context_type: ContextType,
        content: str,
        priority: Priority = Priority.NORMAL,
        relevance: ContextRelevance = ContextRelevance.RELEVANT,
        metadata: Dict[str, Any] = None,
        tags: Set[str] = None,
        source: str = "system",
        duration: float = None,
        success: bool = None,
        related_entries: List[str] = None
    ) -> str:
        """Add context entry with vector embeddings and relevance scoring."""
        if not self.current_session:
            self.create_session()
        
        # Generate semantic hash for deduplication
        semantic_hash = self._generate_semantic_hash(content, context_type)
        
        # Check for duplicates
        if self._is_duplicate_entry(semantic_hash):
            self.logger.debug(f"Skipping duplicate entry: {semantic_hash}")
            return ""
        
        # Calculate tokens and memory impact
        tokens_used = self._estimate_tokens(content)
        memory_impact = len(content.encode('utf-8'))
        
        # Generate embedding
        embedding = self._generate_embedding(content)
        
        # Calculate attention weight based on priority and relevance
        attention_weight = self._calculate_attention_weight(priority, relevance)
        
        # Create entry
        entry = ContextEntry(
            entry_id=str(uuid.uuid4()),
            context_type=context_type,
            content=content,
            priority=priority,
            relevance=relevance,
            metadata=metadata or {},
            tags=tags or set(),
            source=source,
            duration=duration or 0.0,
            success=success if success is not None else True,
            tokens_used=tokens_used,
            memory_impact=memory_impact,
            related_entries=related_entries or [],
            embedding=embedding,
            semantic_hash=semantic_hash,
            attention_weight=attention_weight,
            decay_factor=self.attention_decay_rate
        )
        
        # Add to session
        self.current_session.context_entries.append(entry)
        self.current_session.last_accessed = time.time()
        self.current_session.memory_usage += memory_impact
        self.current_session.token_usage += tokens_used
        self.current_session.current_window_usage += tokens_used
        
        # Save to database
        self._save_entry_to_db(entry)
        
        # Add to vector index
        if embedding is not None:
            self._add_to_vector_index(entry)
        
        # Update analytics
        self._update_analytics(entry)
        
        # Check if compaction is needed
        if self._should_compact():
            self._compact_context()
        
        self.logger.debug(f"Added context entry: {entry.entry_id}")
        return entry.entry_id

    def _generate_semantic_hash(self, content: str, context_type: ContextType) -> str:
        """Generate semantic hash for deduplication."""
        # Normalize content for hashing
        normalized = re.sub(r'\s+', ' ', content.strip().lower())
        hash_input = f"{context_type.value}:{normalized}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _is_duplicate_entry(self, semantic_hash: str) -> bool:
        """Check if entry is a duplicate based on semantic hash."""
        if not self.current_session:
            return False
        
        for entry in self.current_session.context_entries[-100:]:  # Check last 100 entries
            if entry.semantic_hash == semantic_hash:
                return True
        return False

    def _generate_embedding(self, content: str) -> Optional[np.ndarray]:
        """Generate embedding for content."""
        if not self.embedding_model:
            return None
        
        try:
            # Truncate content if too long
            max_length = 512
            if len(content) > max_length:
                content = content[:max_length]
            
            embedding = self.embedding_model.encode([content])[0]
            return embedding.astype(np.float32)
        except Exception as e:
            self.logger.warning(f"Failed to generate embedding: {e}")
            return None

    def _calculate_attention_weight(self, priority: Priority, relevance: ContextRelevance) -> float:
        """Calculate attention weight based on priority and relevance."""
        priority_weights = {
            Priority.CRITICAL: 1.0,
            Priority.HIGH: 0.8,
            Priority.NORMAL: 0.6,
            Priority.LOW: 0.4,
            Priority.ARCHIVE: 0.2
        }
        
        relevance_weights = {
            ContextRelevance.ESSENTIAL: 1.0,
            ContextRelevance.IMPORTANT: 0.8,
            ContextRelevance.RELEVANT: 0.6,
            ContextRelevance.CONTEXTUAL: 0.4,
            ContextRelevance.BACKGROUND: 0.2
        }
        
        return priority_weights[priority] * relevance_weights[relevance]

    def _add_to_vector_index(self, entry: ContextEntry):
        """Add entry to vector index for similarity search."""
        if not self.vector_index or entry.embedding is None:
            return
        
        try:
            with self._vector_lock:
                # Add to FAISS index
                self.vector_index.add(entry.embedding.reshape(1, -1))
                
                # Store mapping
                self.entry_embeddings[entry.entry_id] = entry.embedding
                
                # Save index
                faiss.write_index(self.vector_index, str(self.vector_index_path))
        except Exception as e:
            self.logger.warning(f"Failed to add to vector index: {e}")

    def search_context(
        self,
        query: str,
        context_types: List[ContextType] = None,
        max_results: int = 10,
        min_relevance: float = 0.7
    ) -> List[ContextEntry]:
        """Search context using vector similarity and semantic matching."""
        if not self.current_session:
            return []
        
        results = []
        
        # Vector similarity search
        if self.vector_index and self.embedding_model:
            try:
                query_embedding = self._generate_embedding(query)
                if query_embedding is not None:
                    # Search vector index
                    scores, indices = self.vector_index.search(
                        query_embedding.reshape(1, -1), 
                        min(max_results * 2, len(self.current_session.context_entries))
                    )
                    
                    # Get results
                    for score, idx in zip(scores[0], indices[0]):
                        if score >= min_relevance and idx < len(self.current_session.context_entries):
                            entry = self.current_session.context_entries[idx]
                            if context_types is None or entry.context_type in context_types:
                                results.append(entry)
            except Exception as e:
                self.logger.warning(f"Vector search failed: {e}")
        
        # Fallback to text search
        if not results:
            results = self._text_search_context(query, context_types, max_results)
        
        # Sort by relevance and attention weight
        results.sort(key=lambda x: (x.attention_weight * x.decay_factor), reverse=True)
        
        return results[:max_results]

    def _text_search_context(
        self,
        query: str,
        context_types: List[ContextType] = None,
        max_results: int = 10
    ) -> List[ContextEntry]:
        """Fallback text search for context."""
        if not self.current_session:
            return []
        
        query_lower = query.lower()
        results = []
        
        for entry in reversed(self.current_session.context_entries):  # Most recent first
            if context_types and entry.context_type not in context_types:
                continue
            
            # Simple text matching
            content_lower = entry.content.lower()
            if any(word in content_lower for word in query_lower.split()):
                results.append(entry)
                if len(results) >= max_results:
                    break
        
        return results

    def _should_compact(self) -> bool:
        """Check if context should be compacted."""
        if not self.current_session:
            return False
        
        usage_ratio = self.current_session.current_window_usage / self.current_session.context_window_size
        return usage_ratio >= self.compaction_threshold

    def _compact_context(self):
        """Compact context using Anthropic's best practices."""
        if not self.current_session:
            return
        
        self.logger.info("Starting context compaction")
        
        # Get high-priority entries to keep
        entries_to_keep = []
        entries_to_compact = []
        
        for entry in self.current_session.context_entries:
            if entry.priority in [Priority.CRITICAL, Priority.HIGH]:
                entries_to_keep.append(entry)
            else:
                entries_to_compact.append(entry)
        
        # Create compaction entry
        if entries_to_compact:
            compaction_content = self._create_compaction_summary(entries_to_compact)
            
            compaction_entry = ContextEntry(
                entry_id=str(uuid.uuid4()),
                context_type=ContextType.COMPACTION,
                content=compaction_content,
                priority=Priority.HIGH,
                relevance=ContextRelevance.IMPORTANT,
                metadata={
                    "compacted_entries": len(entries_to_compact),
                    "original_tokens": sum(e.tokens_used for e in entries_to_compact),
                    "compacted_tokens": self._estimate_tokens(compaction_content)
                },
                source="system"
            )
            
            entries_to_keep.append(compaction_entry)
        
        # Update session
        self.current_session.context_entries = entries_to_keep
        self.current_session.compaction_count += 1
        self.current_session.last_compaction = time.time()
        self.current_session.current_window_usage = sum(e.tokens_used for e in entries_to_keep)
        
        # Save updated session
        self._save_session_to_db(self.current_session)
        
        self.logger.info(f"Context compacted: {len(entries_to_compact)} entries -> 1 compaction entry")

    def _create_compaction_summary(self, entries: List[ContextEntry]) -> str:
        """Create intelligent summary of entries for compaction."""
        # Group entries by type
        by_type = defaultdict(list)
        for entry in entries:
            by_type[entry.context_type].append(entry)
        
        summary_parts = []
        
        for context_type, type_entries in by_type.items():
            if context_type == ContextType.TOOL_EXECUTION:
                # Summarize tool executions
                tool_summary = self._summarize_tool_executions(type_entries)
                summary_parts.append(f"Tool Executions: {tool_summary}")
            elif context_type == ContextType.USER_INPUT:
                # Summarize user inputs
                user_summary = self._summarize_user_inputs(type_entries)
                summary_parts.append(f"User Interactions: {user_summary}")
            elif context_type == ContextType.AI_RESPONSE:
                # Summarize AI responses
                ai_summary = self._summarize_ai_responses(type_entries)
                summary_parts.append(f"AI Responses: {ai_summary}")
            else:
                # Generic summary
                summary_parts.append(f"{context_type.value.title()}: {len(type_entries)} entries")
        
        return "\n".join(summary_parts)

    def _summarize_tool_executions(self, entries: List[ContextEntry]) -> str:
        """Summarize tool execution entries."""
        tool_counts = defaultdict(int)
        for entry in entries:
            tool_name = entry.metadata.get("action", "unknown")
            tool_counts[tool_name] += 1
        
        return ", ".join([f"{tool}({count})" for tool, count in tool_counts.items()])

    def _summarize_user_inputs(self, entries: List[ContextEntry]) -> str:
        """Summarize user input entries."""
        if not entries:
            return "None"
        
        # Get key topics from user inputs
        topics = set()
        for entry in entries:
            words = entry.content.lower().split()
            # Extract potential topics (simple heuristic)
            for word in words:
                if len(word) > 4 and word.isalpha():
                    topics.add(word)
        
        return f"{len(entries)} requests covering: {', '.join(list(topics)[:5])}"

    def _summarize_ai_responses(self, entries: List[ContextEntry]) -> str:
        """Summarize AI response entries."""
        return f"{len(entries)} responses with {sum(e.tokens_used for e in entries)} total tokens"

    def add_memory_note(
        self,
        title: str,
        content: str,
        importance: float = 0.5,
        tags: List[str] = None
    ) -> str:
        """Add structured memory note (like Anthropic's memory tool)."""
        note_id = str(uuid.uuid4())
        
        note = {
            "title": title,
            "content": content,
            "tags": tags or [],
            "importance": importance,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "access_count": 0
        }
        
        self.memory_notes[note_id] = note
        
        # Save to database
        self._save_memory_note_to_db(note_id, note)
        
        self.logger.info(f"Added memory note: {title}")
        return note_id

    def get_memory_notes(self, query: str = None, tags: List[str] = None) -> List[Dict[str, Any]]:
        """Retrieve memory notes with optional filtering."""
        notes = list(self.memory_notes.values())
        
        if query:
            query_lower = query.lower()
            notes = [note for note in notes if 
                    query_lower in note["title"].lower() or 
                    query_lower in note["content"].lower()]
        
        if tags:
            notes = [note for note in notes if 
                    any(tag in note["tags"] for tag in tags)]
        
        # Sort by importance and recency
        notes.sort(key=lambda x: (x["importance"], x["last_accessed"]), reverse=True)
        
        return notes

    def _save_entry_to_db(self, entry: ContextEntry):
        """Save context entry to database with vector support."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Serialize embedding
                embedding_blob = None
                if entry.embedding is not None:
                    embedding_blob = entry.embedding.tobytes()
                
                conn.execute("""
                    INSERT OR REPLACE INTO context_entries 
                    (entry_id, session_id, timestamp, context_type, priority, relevance, content, 
                     metadata, tags, source, duration, success, tokens_used, memory_impact, 
                     related_entries, compressed, version, embedding, semantic_hash, 
                     context_window_position, attention_weight, decay_factor, last_accessed, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.entry_id,
                    self.current_session.session_id,
                    entry.timestamp,
                    entry.context_type.value,
                    entry.priority.value,
                    entry.relevance.value,
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
                    entry.version,
                    embedding_blob,
                    entry.semantic_hash,
                    entry.context_window_position,
                    entry.attention_weight,
                    entry.decay_factor,
                    entry.last_accessed,
                    entry.access_count
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
                    (session_id, session_name, created_at, last_accessed, current_directory, 
                     context_summary, context_hash, is_active, session_metadata, performance_metrics,
                     memory_usage, token_usage, compaction_count, last_compaction, 
                     context_window_size, current_window_usage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.session_name,
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
                    session.compaction_count,
                    session.last_compaction,
                    session.context_window_size,
                    session.current_window_usage
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save session to database: {e}")

    def _save_memory_note_to_db(self, note_id: str, note: Dict[str, Any]):
        """Save memory note to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memory_notes 
                    (note_id, session_id, title, content, tags, importance, created_at, last_accessed, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    note_id,
                    self.current_session.session_id if self.current_session else "",
                    note["title"],
                    note["content"],
                    json.dumps(note["tags"]),
                    note["importance"],
                    note["created_at"],
                    note["last_accessed"],
                    note["access_count"]
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save memory note to database: {e}")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4

    def _update_analytics(self, entry: ContextEntry):
        """Update analytics with new entry."""
        self.analytics.total_entries += 1
        self.analytics.entries_by_type[entry.context_type.value] = \
            self.analytics.entries_by_type.get(entry.context_type.value, 0) + 1
        self.analytics.entries_by_priority[entry.priority.value] = \
            self.analytics.entries_by_priority.get(entry.priority.value, 0) + 1
        self.analytics.entries_by_relevance[entry.relevance.value] = \
            self.analytics.entries_by_relevance.get(entry.relevance.value, 0) + 1

    def _optimize_context(self):
        """Optimize context for better performance."""
        if not self.current_session:
            return
        
        # Update attention weights
        self._update_attention_weights()
        
        # Clean up old entries
        self._cleanup_old_entries()
        
        # Optimize vector index
        self._optimize_vector_index()

    def _update_attention_weights(self):
        """Update attention weights based on decay."""
        if not self.current_session:
            return
        
        current_time = time.time()
        for entry in self.current_session.context_entries:
            # Apply decay based on time since last access
            time_since_access = current_time - entry.last_accessed
            decay_factor = entry.decay_factor ** (time_since_access / 3600)  # Decay per hour
            entry.attention_weight *= decay_factor
            entry.last_accessed = current_time

    def _cleanup_old_entries(self):
        """Clean up old, low-relevance entries."""
        if not self.current_session:
            return
        
        # Remove entries with very low attention weights
        threshold = 0.1
        self.current_session.context_entries = [
            entry for entry in self.current_session.context_entries
            if entry.attention_weight > threshold
        ]

    def _optimize_vector_index(self):
        """Optimize vector index for better performance."""
        if not self.vector_index:
            return
        
        try:
            # Rebuild index if it gets too large
            if self.vector_index.ntotal > 10000:
                # Create new index with current entries
                new_index = faiss.IndexFlatIP(384)
                for entry in self.current_session.context_entries:
                    if entry.embedding is not None:
                        new_index.add(entry.embedding.reshape(1, -1))
                
                self.vector_index = new_index
                faiss.write_index(self.vector_index, str(self.vector_index_path))
        except Exception as e:
            self.logger.warning(f"Failed to optimize vector index: {e}")

    def get_context_summary(self) -> str:
        """Get intelligent context summary."""
        if not self.current_session:
            return "No active session"
        
        # Get recent high-priority entries
        recent_entries = [
            entry for entry in self.current_session.context_entries[-20:]
            if entry.priority in [Priority.CRITICAL, Priority.HIGH]
        ]
        
        if not recent_entries:
            return "No significant context available"
        
        # Create summary
        summary_parts = []
        for entry in recent_entries:
            if entry.context_type == ContextType.USER_INPUT:
                summary_parts.append(f"User: {entry.content[:100]}...")
            elif entry.context_type == ContextType.AI_RESPONSE:
                summary_parts.append(f"AI: {entry.content[:100]}...")
            elif entry.context_type == ContextType.TOOL_EXECUTION:
                action = entry.metadata.get("action", "unknown")
                summary_parts.append(f"Tool: {action}")
        
        return "\n".join(summary_parts)

    def get_recent_context(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent context entries for AI system compatibility."""
        if not self.current_session:
            return []
        
        recent_entries = self.current_session.context_entries[-limit:]
        context_list = []
        
        for entry in recent_entries:
            context_list.append({
                "type": entry.context_type.value,
                "content": entry.content,
                "timestamp": entry.timestamp,
                "priority": entry.priority.value,
                "relevance": entry.relevance.value,
                "metadata": entry.metadata,
                "tags": list(entry.tags),
                "source": entry.source,
                "success": entry.success,
                "tokens_used": entry.tokens_used,
                "attention_weight": entry.attention_weight
            })
        
        return context_list

    def get_context_by_type(self, context_type: ContextType, limit: int = 10) -> List[Dict[str, Any]]:
        """Get context entries by type."""
        if not self.current_session:
            return []
        
        filtered_entries = [
            entry for entry in self.current_session.context_entries
            if entry.context_type == context_type
        ][-limit:]
        
        context_list = []
        for entry in filtered_entries:
            context_list.append({
                "type": entry.context_type.value,
                "content": entry.content,
                "timestamp": entry.timestamp,
                "priority": entry.priority.value,
                "relevance": entry.relevance.value,
                "metadata": entry.metadata,
                "tags": list(entry.tags),
                "source": entry.source,
                "success": entry.success,
                "tokens_used": entry.tokens_used,
                "attention_weight": entry.attention_weight
            })
        
        return context_list

    def get_context_by_priority(self, priority: Priority, limit: int = 10) -> List[Dict[str, Any]]:
        """Get context entries by priority."""
        if not self.current_session:
            return []
        
        filtered_entries = [
            entry for entry in self.current_session.context_entries
            if entry.priority == priority
        ][-limit:]
        
        context_list = []
        for entry in filtered_entries:
            context_list.append({
                "type": entry.context_type.value,
                "content": entry.content,
                "timestamp": entry.timestamp,
                "priority": entry.priority.value,
                "relevance": entry.relevance.value,
                "metadata": entry.metadata,
                "tags": list(entry.tags),
                "source": entry.source,
                "success": entry.success,
                "tokens_used": entry.tokens_used,
                "attention_weight": entry.attention_weight
            })
        
        return context_list

    def get_context_by_hour(self, hour: int) -> List[Dict[str, Any]]:
        """Get context entries by hour of day."""
        if not self.current_session:
            return []
        
        from datetime import datetime
        current_time = datetime.now()
        target_hour = current_time.replace(hour=hour, minute=0, second=0, microsecond=0)
        target_timestamp = target_hour.timestamp()
        
        filtered_entries = [
            entry for entry in self.current_session.context_entries
            if datetime.fromtimestamp(entry.timestamp).hour == hour
        ]
        
        context_list = []
        for entry in filtered_entries:
            context_list.append({
                "type": entry.context_type.value,
                "content": entry.content,
                "timestamp": entry.timestamp,
                "priority": entry.priority.value,
                "relevance": entry.relevance.value,
                "metadata": entry.metadata,
                "tags": list(entry.tags),
                "source": entry.source,
                "success": entry.success,
                "tokens_used": entry.tokens_used,
                "attention_weight": entry.attention_weight
            })
        
        return context_list

    def search_context_by_time(self, time_range: str) -> List[Dict[str, Any]]:
        """Search context by time range."""
        if not self.current_session:
            return []
        
        # Simple time range parsing
        if time_range == "today":
            from datetime import datetime, timedelta
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            start_timestamp = today.timestamp()
        elif time_range == "yesterday":
            from datetime import datetime, timedelta
            yesterday = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            start_timestamp = yesterday.timestamp()
        else:
            # Default to last 24 hours
            start_timestamp = time.time() - 24 * 3600
        
        filtered_entries = [
            entry for entry in self.current_session.context_entries
            if entry.timestamp >= start_timestamp
        ]
        
        context_list = []
        for entry in filtered_entries:
            context_list.append({
                "type": entry.context_type.value,
                "content": entry.content,
                "timestamp": entry.timestamp,
                "priority": entry.priority.value,
                "relevance": entry.relevance.value,
                "metadata": entry.metadata,
                "tags": list(entry.tags),
                "source": entry.source,
                "success": entry.success,
                "tokens_used": entry.tokens_used,
                "attention_weight": entry.attention_weight
            })
        
        return context_list

    def get_active_session(self) -> Optional[SessionContext]:
        """Get the currently active session."""
        return self.current_session

    def set_active_session(self, session_id: str) -> bool:
        """Set the active session."""
        if session_id in self.sessions:
            self.current_session = self.sessions[session_id]
            self.current_session.last_accessed = time.time()
            self._save_session_to_db(self.current_session)
            return True
        return False

    def get_session_list(self) -> List[Dict[str, Any]]:
        """Get list of all sessions."""
        sessions = []
        for session_id, session in self.sessions.items():
            sessions.append({
                "session_id": session_id,
                "session_name": session.session_name,
                "created_at": session.created_at,
                "last_accessed": session.last_accessed,
                "entry_count": len(session.context_entries),
                "memory_usage": session.memory_usage,
                "token_usage": session.token_usage,
                "compaction_count": session.compaction_count
            })
        return sorted(sessions, key=lambda x: x["last_accessed"], reverse=True)

    def cleanup_old_sessions(self, max_age_days: int = 30):
        """Clean up old sessions."""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        sessions_to_remove = []
        for session_id, session in self.sessions.items():
            if current_time - session.last_accessed > max_age_seconds:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
        
        if sessions_to_remove:
            self.logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")

    def shutdown(self):
        """Shutdown the context manager."""
        try:
            # Save all sessions
            for session in self.sessions.values():
                self._save_session_to_db(session)
            
            # Save memory notes
            for note_id, note in self.memory_notes.items():
                self._save_memory_note_to_db(note_id, note)
            
            # Shutdown executor
            self._executor.shutdown(wait=True)
            
            self.logger.info("Context manager shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Convenience functions for backward compatibility
def create_context_manager() -> AdvancedContextManager:
    """Create a new advanced context manager instance."""
    return AdvancedContextManager()