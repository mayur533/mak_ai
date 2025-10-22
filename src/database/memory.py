"""
Database management for the AI Assistant System.
Handles memory storage, tool persistence, and data management.
"""

import json
import sqlite3
import time
import uuid
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.config.settings import settings
from src.logging.logger import logger


@dataclass
class MemoryItem:
    """Represents a memory item in the system."""

    id: str
    content: str
    timestamp: float
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class Tool:
    """Represents a tool in the system."""

    name: str
    code: str
    doc: str
    is_dynamic: bool = False
    last_used: float = 0
    func: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "code": self.code,
            "doc": self.doc,
            "is_dynamic": self.is_dynamic,
            "last_used": self.last_used,
        }


class MemoryManager:
    """Manages memory storage and retrieval for the AI system."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize memory manager with database connection."""
        self.db_path = db_path or settings.DB_PATH
        self.memory = deque(maxlen=settings.MAX_HISTORY)
        self.db = None
        self._init_db()
        self._load_memory()

    def _init_db(self):
        """Initialize the SQLite database and create necessary tables."""
        logger.info("Initializing database...")

        # Ensure database directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db = sqlite3.connect(self.db_path)
        cursor = self.db.cursor()

        # Create memory table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS memory (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            timestamp REAL NOT NULL,
            metadata TEXT
        )"""
        )

        # Create tools table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS tools (
            name TEXT PRIMARY KEY,
            code TEXT NOT NULL,
            doc TEXT,
            is_dynamic INTEGER DEFAULT 0,
            last_used REAL DEFAULT 0
        )"""
        )

        # Create execution history table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS execution_history (
            id TEXT PRIMARY KEY,
            step_id TEXT NOT NULL,
            step_description TEXT,
            result TEXT,
            timestamp REAL NOT NULL,
            error_message TEXT
        )"""
        )

        # Create tool metadata table for enhanced tool information
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS tool_metadata (
            name TEXT PRIMARY KEY,
            category TEXT,
            parameters TEXT,
            examples TEXT,
            result_formats TEXT,
            usage_count INTEGER DEFAULT 0,
            registered_at REAL,
            last_updated REAL,
            FOREIGN KEY (name) REFERENCES tools (name)
        )"""
        )

        # Create indexes for better query performance
        logger.info("Creating database indexes for optimized performance...")
        
        # Index on memory timestamp for faster recent memory queries
        cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_memory_timestamp 
            ON memory(timestamp DESC)"""
        )
        
        # Index on tool last_used for tool usage tracking
        cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_tools_last_used 
            ON tools(last_used DESC)"""
        )
        
        # Index on execution history timestamp
        cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_execution_timestamp 
            ON execution_history(timestamp DESC)"""
        )
        
        # Index on tool_metadata category for filtering
        cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_tool_metadata_category 
            ON tool_metadata(category)"""
        )
        
        # Index on tool_metadata usage_count for popular tools
        cursor.execute(
            """CREATE INDEX IF NOT EXISTS idx_tool_metadata_usage 
            ON tool_metadata(usage_count DESC)"""
        )

        self.db.commit()
        logger.success("Database initialized successfully with optimized indexes.")

    def _load_memory(self):
        """Load memory items from database into memory."""
        logger.info("Loading memory from database...")
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT id, content, timestamp, metadata 
            FROM memory 
            ORDER BY timestamp DESC 
            LIMIT ?
        """,
            (settings.MAX_HISTORY,),
        )

        loaded_count = 0
        for mem_id, content, timestamp, metadata in cursor.fetchall():
            try:
                item = MemoryItem(
                    id=mem_id,
                    content=content,
                    timestamp=timestamp,
                    metadata=json.loads(metadata) if metadata else {},
                )
                self.memory.append(item)
                loaded_count += 1
            except Exception as e:
                logger.error(f"Error loading memory item {mem_id}: {e}")

        logger.success(f"Loaded {loaded_count} memory items.")

    def remember(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Store a new memory item."""
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        self.memory.append(item)

        # Store in database
        cursor = self.db.cursor()
        cursor.execute(
            """
            INSERT INTO memory (id, content, timestamp, metadata)
            VALUES (?, ?, ?, ?)
        """,
            (item.id, item.content, item.timestamp, json.dumps(item.metadata)),
        )
        self.db.commit()

        logger.debug(f"Remembered: {content[:50]}...")
        return item.id

    def recall(self, query: str, top_k: int = 5) -> List[MemoryItem]:
        """Recall relevant memories based on query."""
        relevant_memories = [
            item for item in self.memory if query.lower() in item.content.lower()
        ]

        # Sort by relevance (query frequency) and recency
        relevant_memories.sort(
            key=lambda x: (x.content.lower().count(query.lower()), x.timestamp),
            reverse=True,
        )

        return relevant_memories[:top_k]

    def get_recent_memories(self, count: int = 10) -> List[MemoryItem]:
        """Get the most recent memories."""
        return list(self.memory)[-count:]

    def clear_memory(self):
        """Clear all memory items."""
        self.memory.clear()
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM memory")
        self.db.commit()
        logger.info("Memory cleared.")

    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()
            logger.info("Database connection closed.")


class ToolManager:
    """Manages tool storage and retrieval."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize tool manager with database connection."""
        self.db_path = db_path or settings.DB_PATH
        self.tools: Dict[str, Tool] = {}
        self.db = None
        self._init_db()
        self._load_tools()

    def _init_db(self):
        """Initialize database connection."""
        self.db = sqlite3.connect(self.db_path)
        logger.info("Tool manager database initialized.")

    def _load_tools(self):
        """Load tools from database."""
        logger.info("Loading tools from database...")
        cursor = self.db.cursor()
        cursor.execute("SELECT name, code, doc, is_dynamic, last_used FROM tools")

        loaded_count = 0
        for name, code, doc, is_dynamic, last_used in cursor.fetchall():
            try:
                tool = Tool(
                    name=name,
                    code=code,
                    doc=doc,
                    is_dynamic=bool(is_dynamic),
                    last_used=last_used,
                )
                self.tools[name] = tool
                loaded_count += 1
            except Exception as e:
                logger.error(f"Error loading tool '{name}': {e}")

        logger.success(f"Loaded {loaded_count} tools from database.")

    def register_tool(self, tool: Tool):
        """Register a new tool."""
        self.tools[tool.name] = tool

        # Store in database
        cursor = self.db.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO tools (name, code, doc, is_dynamic, last_used)
            VALUES (?, ?, ?, ?, ?)
        """,
            (tool.name, tool.code, tool.doc, int(tool.is_dynamic), tool.last_used),
        )
        self.db.commit()

        logger.success(f"Tool '{tool.name}' registered successfully.")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def tool_exists(self, name: str) -> bool:
        """Check if a tool exists."""
        return name in self.tools

    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self.tools.keys())

    def update_tool_usage(self, name: str):
        """Update tool last used timestamp."""
        if name in self.tools:
            self.tools[name].last_used = time.time()
            cursor = self.db.cursor()
            cursor.execute(
                "UPDATE tools SET last_used = ? WHERE name = ?", (time.time(), name)
            )
            self.db.commit()

    def save_tool_metadata(self, name: str, metadata: Dict[str, Any]):
        """Save enhanced tool metadata to database."""
        try:
            cursor = self.db.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO tool_metadata 
                (name, category, parameters, examples, result_formats, usage_count, registered_at, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    metadata.get("category", "general"),
                    json.dumps(metadata.get("parameters", [])),
                    json.dumps(metadata.get("examples", [])),
                    json.dumps(metadata.get("result_formats", {})),
                    metadata.get("usage_count", 0),
                    metadata.get("registered_at", time.time()),
                    time.time()
                )
            )
            self.db.commit()
            logger.debug(f"Saved metadata for tool '{name}'")
        except Exception as e:
            logger.error(f"Failed to save metadata for tool '{name}': {e}")

    def get_tool_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get enhanced tool metadata from database."""
        try:
            cursor = self.db.cursor()
            cursor.execute(
                "SELECT category, parameters, examples, result_formats, usage_count, registered_at, last_updated FROM tool_metadata WHERE name = ?",
                (name,)
            )
            result = cursor.fetchone()
            
            if result:
                return {
                    "name": name,
                    "category": result[0],
                    "parameters": json.loads(result[1]) if result[1] else [],
                    "examples": json.loads(result[2]) if result[2] else [],
                    "result_formats": json.loads(result[3]) if result[3] else {},
                    "usage_count": result[4],
                    "registered_at": result[5],
                    "last_updated": result[6]
                }
        except Exception as e:
            logger.error(f"Failed to get metadata for tool '{name}': {e}")
        
        return None

    def get_all_tool_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all tools."""
        try:
            cursor = self.db.cursor()
            cursor.execute(
                "SELECT name, category, parameters, examples, result_formats, usage_count, registered_at, last_updated FROM tool_metadata"
            )
            
            metadata_dict = {}
            for row in cursor.fetchall():
                name = row[0]
                metadata_dict[name] = {
                    "name": name,
                    "category": row[1],
                    "parameters": json.loads(row[2]) if row[2] else [],
                    "examples": json.loads(row[3]) if row[3] else [],
                    "result_formats": json.loads(row[4]) if row[4] else {},
                    "usage_count": row[5],
                    "registered_at": row[6],
                    "last_updated": row[7]
                }
            
            return metadata_dict
        except Exception as e:
            logger.error(f"Failed to get all tool metadata: {e}")
            return {}

    def remove_tool(self, name: str) -> bool:
        """Remove a tool."""
        if name in self.tools:
            del self.tools[name]
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM tools WHERE name = ?", (name,))
            cursor.execute("DELETE FROM tool_metadata WHERE name = ?", (name,))
            self.db.commit()
            logger.info(f"Tool '{name}' removed.")
            return True
        return False

    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()
            logger.info("Tool manager database connection closed.")


class ExecutionHistory:
    """Manages execution history for the AI system."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize execution history manager."""
        self.db_path = db_path or settings.DB_PATH
        self.db = None
        self._init_db()

    def _init_db(self):
        """Initialize database connection."""
        self.db = sqlite3.connect(self.db_path)
        logger.info("Execution history manager initialized.")

    def log_execution(
        self,
        step_id: str,
        step_description: str,
        result: str,
        error_message: str = None,
    ):
        """Log an execution step."""
        cursor = self.db.cursor()
        cursor.execute(
            """
            INSERT INTO execution_history 
            (id, step_id, step_description, result, timestamp, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                str(uuid.uuid4()),
                step_id,
                step_description,
                result,
                time.time(),
                error_message,
            ),
        )
        self.db.commit()

    def get_failed_steps(self) -> List[Dict[str, Any]]:
        """Get all failed execution steps."""
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT step_id, step_description, error_message, timestamp
            FROM execution_history 
            WHERE result = 'failed'
            ORDER BY timestamp DESC
        """
        )

        return [
            {
                "step_id": row[0],
                "step_description": row[1],
                "error_message": row[2],
                "timestamp": row[3],
            }
            for row in cursor.fetchall()
        ]

    def has_failed_before(self, step_id: str) -> bool:
        """Check if a step has failed before."""
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) FROM execution_history 
            WHERE step_id = ? AND result = 'failed'
        """,
            (step_id,),
        )

        return cursor.fetchone()[0] > 0

    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()
            logger.info("Execution history database connection closed.")
