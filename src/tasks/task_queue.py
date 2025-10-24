"""
Background Task Queue System for AI Assistant.
Provides both simple threading-based queue and optional Celery/RQ integration.
"""

import json
import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import traceback

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config.settings import settings
from src.logging.logger import logger


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a background task."""
    task_id: str
    name: str
    func_name: str
    args: tuple
    kwargs: dict
    status: TaskStatus
    priority: TaskPriority
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "func_name": self.func_name,
            "args": self.args,
            "kwargs": self.kwargs,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": str(self.result) if self.result else None,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }


class SimpleTaskQueue:
    """
    Simple thread-based task queue for background job processing.
    No external dependencies required.
    """
    
    def __init__(self, num_workers: int = 4, db_path: Optional[str] = None):
        """Initialize the task queue."""
        self.num_workers = num_workers
        self.db_path = db_path or str(settings.BASE_DIR / "db" / "tasks.db")
        self.task_queue = queue.PriorityQueue()
        self.workers: List[threading.Thread] = []
        self.running = False
        self.registered_functions: Dict[str, Callable] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.lock = threading.RLock()
        
        # Initialize database
        self._init_db()
        
        # Load pending tasks from database
        self._load_pending_tasks()
        
        logger.info(f"SimpleTaskQueue initialized with {num_workers} workers")
    
    def _init_db(self):
        """Initialize the task database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                func_name TEXT NOT NULL,
                args TEXT,
                kwargs TEXT,
                status TEXT NOT NULL,
                priority INTEGER NOT NULL,
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                result TEXT,
                error TEXT,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3
            )
        """)
        
        # Create indexes for better performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_status 
            ON tasks(status)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_priority 
            ON tasks(priority DESC, created_at ASC)
        """)
        
        conn.commit()
        conn.close()
        
        logger.success("Task database initialized")
    
    def _load_pending_tasks(self):
        """Load pending tasks from database on startup."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT task_id, name, func_name, args, kwargs, status, priority,
                   created_at, started_at, completed_at, result, error, 
                   retry_count, max_retries
            FROM tasks
            WHERE status IN ('pending', 'retry')
            ORDER BY priority DESC, created_at ASC
        """)
        
        loaded_count = 0
        for row in cursor.fetchall():
            try:
                task = Task(
                    task_id=row[0],
                    name=row[1],
                    func_name=row[2],
                    args=json.loads(row[3]) if row[3] else (),
                    kwargs=json.loads(row[4]) if row[4] else {},
                    status=TaskStatus(row[5]),
                    priority=TaskPriority(row[6]),
                    created_at=row[7],
                    started_at=row[8],
                    completed_at=row[9],
                    result=row[10],
                    error=row[11],
                    retry_count=row[12],
                    max_retries=row[13],
                )
                # Re-queue the task
                self.task_queue.put((task.priority.value * -1, task.created_at, task))
                loaded_count += 1
            except Exception as e:
                logger.error(f"Error loading task {row[0]}: {e}")
        
        conn.close()
        
        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} pending tasks from database")
    
    def register_function(self, name: str, func: Callable):
        """Register a function that can be called as a background task."""
        self.registered_functions[name] = func
        logger.debug(f"Registered function: {name}")
    
    def add_task(
        self,
        name: str,
        func_name: str,
        args: tuple = (),
        kwargs: dict = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
    ) -> str:
        """
        Add a new task to the queue.
        
        Args:
            name: Human-readable task name
            func_name: Name of registered function to call
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            priority: Task priority
            max_retries: Maximum number of retry attempts
            
        Returns:
            Task ID
        """
        if func_name not in self.registered_functions:
            raise ValueError(f"Function '{func_name}' not registered. Use register_function() first.")
        
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        task = Task(
            task_id=task_id,
            name=name,
            func_name=func_name,
            args=args,
            kwargs=kwargs or {},
            status=TaskStatus.PENDING,
            priority=priority,
            created_at=time.time(),
            max_retries=max_retries,
        )
        
        # Save to database
        self._save_task(task)
        
        # Add to queue (priority is negative for max heap behavior)
        self.task_queue.put((priority.value * -1, task.created_at, task))
        
        logger.info(f"Added task '{name}' (ID: {task_id}) with priority {priority.name}")
        return task_id
    
    def _save_task(self, task: Task):
        """Save task to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO tasks
            (task_id, name, func_name, args, kwargs, status, priority,
             created_at, started_at, completed_at, result, error, retry_count, max_retries)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id,
            task.name,
            task.func_name,
            json.dumps(task.args, default=str),
            json.dumps(task.kwargs, default=str),
            task.status.value,
            task.priority.value,
            task.created_at,
            task.started_at,
            task.completed_at,
            json.dumps(task.result, default=str) if task.result else None,
            task.error,
            task.retry_count,
            task.max_retries,
        ))
        
        conn.commit()
        conn.close()
    
    def _worker(self):
        """Worker thread that processes tasks from the queue."""
        worker_id = threading.current_thread().name
        logger.debug(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get task with timeout to allow checking running flag
                try:
                    priority, created_at, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the task
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()
                
                with self.lock:
                    self.active_tasks[task.task_id] = task
                
                self._save_task(task)
                logger.info(f"[{worker_id}] Processing task '{task.name}' (ID: {task.task_id})")
                
                try:
                    # Execute the registered function
                    func = self.registered_functions[task.func_name]
                    result = func(*task.args, **task.kwargs)
                    
                    # Task completed successfully
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = time.time()
                    
                    logger.success(
                        f"[{worker_id}] Task '{task.name}' completed in "
                        f"{task.completed_at - task.started_at:.2f}s"
                    )
                    
                except Exception as e:
                    # Task failed
                    task.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                    
                    # Retry if allowed
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        task.status = TaskStatus.RETRY
                        logger.warning(
                            f"[{worker_id}] Task '{task.name}' failed, "
                            f"retrying ({task.retry_count}/{task.max_retries}): {e}"
                        )
                        # Re-queue for retry with slightly lower priority
                        self.task_queue.put((task.priority.value * -1 + 1, time.time(), task))
                    else:
                        task.status = TaskStatus.FAILED
                        task.completed_at = time.time()
                        logger.error(f"[{worker_id}] Task '{task.name}' failed after {task.max_retries} retries: {e}")
                
                # Save final state
                self._save_task(task)
                
                with self.lock:
                    if task.task_id in self.active_tasks:
                        del self.active_tasks[task.task_id]
                
                # Mark queue task as done
                self.task_queue.task_done()
                
            except Exception as e:
                logger.error(f"[{worker_id}] Worker error: {e}")
                traceback.print_exc()
        
        logger.debug(f"Worker {worker_id} stopped")
    
    def start(self):
        """Start the task queue workers."""
        if self.running:
            logger.warning("Task queue already running")
            return
        
        self.running = True
        
        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker,
                name=f"TaskWorker-{i+1}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.success(f"Task queue started with {self.num_workers} workers")
    
    def stop(self, timeout: int = 30):
        """Stop the task queue and wait for workers to finish."""
        if not self.running:
            return
        
        logger.info("Stopping task queue...")
        self.running = False
        
        # Wait for all workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout / self.num_workers)
        
        self.workers.clear()
        logger.success("Task queue stopped")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific task."""
        # Check active tasks first
        with self.lock:
            if task_id in self.active_tasks:
                return self.active_tasks[task_id].to_dict()
        
        # Check database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT task_id, name, func_name, args, kwargs, status, priority,
                   created_at, started_at, completed_at, result, error,
                   retry_count, max_retries
            FROM tasks
            WHERE task_id = ?
        """, (task_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            task = Task(
                task_id=row[0],
                name=row[1],
                func_name=row[2],
                args=json.loads(row[3]) if row[3] else (),
                kwargs=json.loads(row[4]) if row[4] else {},
                status=TaskStatus(row[5]),
                priority=TaskPriority(row[6]),
                created_at=row[7],
                started_at=row[8],
                completed_at=row[9],
                result=row[10],
                error=row[11],
                retry_count=row[12],
                max_retries=row[13],
            )
            return task.to_dict()
        
        return None
    
    def get_all_tasks(self, status: Optional[TaskStatus] = None) -> List[Dict[str, Any]]:
        """Get all tasks, optionally filtered by status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if status:
            cursor.execute("""
                SELECT task_id, name, func_name, args, kwargs, status, priority,
                       created_at, started_at, completed_at, result, error,
                       retry_count, max_retries
                FROM tasks
                WHERE status = ?
                ORDER BY created_at DESC
            """, (status.value,))
        else:
            cursor.execute("""
                SELECT task_id, name, func_name, args, kwargs, status, priority,
                       created_at, started_at, completed_at, result, error,
                       retry_count, max_retries
                FROM tasks
                ORDER BY created_at DESC
            """)
        
        tasks = []
        for row in cursor.fetchall():
            task = Task(
                task_id=row[0],
                name=row[1],
                func_name=row[2],
                args=json.loads(row[3]) if row[3] else (),
                kwargs=json.loads(row[4]) if row[4] else {},
                status=TaskStatus(row[5]),
                priority=TaskPriority(row[6]),
                created_at=row[7],
                started_at=row[8],
                completed_at=row[9],
                result=row[10],
                error=row[11],
                retry_count=row[12],
                max_retries=row[13],
            )
            tasks.append(task.to_dict())
        
        conn.close()
        return tasks
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        # Can only cancel pending tasks, not running ones
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE tasks
            SET status = 'cancelled'
            WHERE task_id = ? AND status = 'pending'
        """, (task_id,))
        
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        if rows_affected > 0:
            logger.info(f"Cancelled task {task_id}")
            return True
        
        return False
    
    def get_queue_size(self) -> int:
        """Get the number of pending tasks in the queue."""
        return self.task_queue.qsize()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get task queue statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Count by status
        cursor.execute("""
            SELECT status, COUNT(*) 
            FROM tasks 
            GROUP BY status
        """)
        
        status_counts = {row[0]: row[1] for row in cursor.fetchall()}
        stats['by_status'] = status_counts
        
        # Average execution time for completed tasks
        cursor.execute("""
            SELECT AVG(completed_at - started_at) as avg_time
            FROM tasks
            WHERE status = 'completed' AND started_at IS NOT NULL
        """)
        
        avg_time = cursor.fetchone()[0]
        stats['avg_execution_time'] = avg_time if avg_time else 0
        
        # Total tasks
        cursor.execute("SELECT COUNT(*) FROM tasks")
        stats['total_tasks'] = cursor.fetchone()[0]
        
        # Active tasks
        with self.lock:
            stats['active_tasks'] = len(self.active_tasks)
        
        stats['queue_size'] = self.get_queue_size()
        stats['num_workers'] = self.num_workers
        stats['is_running'] = self.running
        
        conn.close()
        return stats


# Global task queue instance
_global_task_queue: Optional[SimpleTaskQueue] = None


def get_task_queue(num_workers: int = 4) -> SimpleTaskQueue:
    """Get or create the global task queue instance."""
    global _global_task_queue
    
    if _global_task_queue is None:
        _global_task_queue = SimpleTaskQueue(num_workers=num_workers)
        _global_task_queue.start()
    
    return _global_task_queue


def shutdown_task_queue():
    """Shutdown the global task queue."""
    global _global_task_queue
    
    if _global_task_queue:
        _global_task_queue.stop()
        _global_task_queue = None





