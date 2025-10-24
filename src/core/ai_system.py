"""
Core AI System for the AI Assistant System.
Main orchestrator that handles AI interactions, tool management, and task execution.
"""

import asyncio
import json
import os
import re
import time
import hashlib
from collections import deque
from typing import Dict, Any, List, Optional, Annotated, Sequence, TypedDict
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import requests
import aiohttp
import aiofiles

# Advanced AI Frameworks
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from crewai import Agent, Task, Crew, Process
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import BaseModel, Field

# Add parent directory to path for imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config.settings import settings
from src.logging.logger import Logger
from src.database.memory import MemoryManager, ToolManager, ExecutionHistory, Tool
from src.tools.base_tools import BaseTools
from src.tools.voice_tools import VoiceTools
from src.core.context_manager import AdvancedContextManager
from src.tools.google_search import GoogleSearchTool
from src.core.gemini_client import GeminiClient
from src.monitoring import metrics_collector, record_request_metric, record_tool_metric
from src.api import start_health_server
from src.tasks.task_queue import get_task_queue, shutdown_task_queue, TaskPriority


# Autonomous System State Management
class AutonomousState(TypedDict):
    """State for the autonomous AI system using LangGraph pattern."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_task: Optional[str]
    task_history: List[Dict[str, Any]]
    error_count: int
    success_count: int
    learning_data: Dict[str, Any]
    self_healing_active: bool
    agent_rotation_count: int


# Self-Healing and Learning Components
class SelfHealingSystem:
    """Handles automatic error recovery and learning."""
    
    def __init__(self):
        self.error_patterns = {}
        self.success_patterns = {}
        self.recovery_strategies = {}
        self.learning_threshold = 3
        
    def analyze_error(self, error: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error patterns and suggest recovery strategies."""
        error_signature = hashlib.md5(error.encode()).hexdigest()[:8]
        
        if error_signature not in self.error_patterns:
            self.error_patterns[error_signature] = {
                'count': 0,
                'contexts': [],
                'recovery_attempts': []
            }
        
        self.error_patterns[error_signature]['count'] += 1
        self.error_patterns[error_signature]['contexts'].append(context)
        
        return {
            'error_signature': error_signature,
            'frequency': self.error_patterns[error_signature]['count'],
            'suggested_recovery': self._suggest_recovery(error_signature, context)
        }
    
    def _suggest_recovery(self, error_signature: str, context: Dict[str, Any]) -> str:
        """Suggest recovery strategy based on error patterns."""
        if self.error_patterns[error_signature]['count'] > self.learning_threshold:
            return "Use alternative approach - this method has failed multiple times"
        return "Retry with exponential backoff"


class AutonomousAISystem:
    """
    Autonomous AI System with self-healing, learning, and multi-agent capabilities.
    Uses LangGraph for ReAct pattern, CrewAI for multi-agent collaboration, and LlamaIndex for intelligent processing.
    """

    def __init__(self, voice_mode: bool = None, max_workers: int = 4):
        """Initialize the AI system."""
        self.voice_mode = (
            voice_mode if voice_mode is not None else settings.VOICE_ENABLED
        )
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.session = None
        self.semaphore = asyncio.Semaphore(max_workers)
        self.logger = Logger("ai_system", clean_console=True)

        # Initialize components
        self.memory_manager = MemoryManager()
        self.tool_manager = ToolManager()
        self.execution_history = ExecutionHistory()
        self.context_manager = AdvancedContextManager()
        self.google_search = GoogleSearchTool()
        self.gemini_client = GeminiClient(api_key=settings.GEMINI_API_KEY)

        # Load system configuration
        self.system_config = self._load_system_config()

        # Initialize tools
        self.base_tools = BaseTools(system=self)
        self.voice_tools = VoiceTools(system=self)
        
        # Initialize background task queue
        self.task_queue = get_task_queue(num_workers=4)
        self.logger.info("Background task queue initialized")

        # Autonomous system context - completely dynamic and learning
        self.context = {
            "cwd": os.getcwd(),
            "os": sys.platform,
            "python_version": sys.version,
            "system_info": {},
            "learning_mode": True,
            "self_healing_enabled": True,
            "agent_rotation_enabled": True,
            "execution_history": deque(maxlen=50),  # Increased for better learning
            "conversation_history": deque(maxlen=50),
            "error_patterns": {},
            "success_patterns": {},
        }

        self.active = True
        
        # Agent management for preventing loops
        self.agent_usage_history = deque(maxlen=10)  # Track last 10 agent uses
        self.current_agent = None
        self.agent_rotation_enabled = True
        
        # Error tracking for loop detection
        self.recent_errors = deque(maxlen=10)  # Track last 10 errors
        self.error_retry_counts = {}  # Track retry counts per error signature

        # Register core tools
        self._register_core_tools()
        self._load_tools_from_files()
        
        # Refresh tools from database (handles database recreation)
        self._refresh_tools_from_database()
        
        self._get_initial_system_details()

    def _setup_llamaindex(self):
        """Initialize LlamaIndex for intelligent document processing."""
        try:
            # Configure LlamaIndex settings
            Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
            Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
            
            # Create knowledge base directory if it doesn't exist
            kb_dir = Path("knowledge_base")
            kb_dir.mkdir(exist_ok=True)
            
            # Initialize vector store index
            self.vector_index = VectorStoreIndex.from_documents([])
            self.logger.info("LlamaIndex initialized for document processing")
            
        except Exception as e:
            self.logger.warning(f"LlamaIndex setup failed: {e}. Continuing without document processing.")

    def _create_dynamic_agents(self):
        """Create dynamic CrewAI agents based on task requirements."""
        try:
            # Base agent template
            base_agent_config = {
                "llm": self.llm,
                "verbose": True,
                "allow_delegation": True,
                "max_iter": 3,
            }
            
            # Create specialized agents dynamically
            self.crew_agents = {
                "coordinator": Agent(
                    role="Task Coordinator",
                    goal="Analyze tasks and coordinate execution",
                    backstory="Expert at breaking down complex tasks and coordinating multiple agents",
                    **base_agent_config
                ),
                "executor": Agent(
                    role="Task Executor", 
                    goal="Execute specific tasks and operations",
                    backstory="Specialized in executing various system operations and tool usage",
                    **base_agent_config
                ),
                "analyzer": Agent(
                    role="Data Analyzer",
                    goal="Analyze data, documents, and results",
                    backstory="Expert at processing and analyzing various types of data",
                    **base_agent_config
                ),
                "debugger": Agent(
                    role="Error Debugger",
                    goal="Identify and fix errors automatically",
                    backstory="Specialized in error analysis and automatic problem resolution",
                    **base_agent_config
                )
            }
            
            self.logger.info("Dynamic CrewAI agents created successfully")
            
        except Exception as e:
            self.logger.warning(f"CrewAI agent creation failed: {e}. Using fallback mode.")

    def _create_react_workflow(self):
        """Create LangGraph ReAct workflow for autonomous operation."""
        try:
            # Define the workflow
            workflow = StateGraph(AutonomousState)
            
            # Add nodes
            workflow.add_node("agent", self._call_agent)
            workflow.add_node("tools", self._call_tools)
            
            # Set entry point
            workflow.set_entry_point("agent")
            
            # Add conditional edges
            workflow.add_conditional_edges(
                "agent",
                self._should_continue,
                {
                    "continue": "tools",
                    "end": END,
                }
            )
            
            # Add edge back to agent
            workflow.add_edge("tools", "agent")
            
            # Compile workflow
            return workflow.compile()
            
        except Exception as e:
            self.logger.warning(f"LangGraph workflow creation failed: {e}. Using fallback mode.")
            return None

    def _call_agent(self, state: AutonomousState):
        """Call the AI agent with current state."""
        try:
            # Get the latest message
            messages = state["messages"]
            if not messages:
                return {"messages": [AIMessage(content="Hello! How can I help you today?")]}
            
            # Use the LLM to generate response
            response = self.llm.invoke(messages)
            return {"messages": [response]}
            
        except Exception as e:
            self.logger.error(f"Agent call failed: {e}")
            return {"messages": [AIMessage(content=f"I encountered an error: {e}")]}

    def _call_tools(self, state: AutonomousState):
        """Execute tools based on agent's tool calls."""
        try:
            messages = state["messages"]
            last_message = messages[-1]
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_outputs = []
                for tool_call in last_message.tool_calls:
                    # Execute the tool
                    result = self._execute_tool_call(tool_call)
                    tool_outputs.append(
                        ToolMessage(
                            content=str(result),
                            name=tool_call["name"],
                            tool_call_id=tool_call["id"]
                        )
                    )
                return {"messages": tool_outputs}
            
            return {"messages": []}
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {e}")
            return {"messages": [ToolMessage(content=f"Tool execution error: {e}", name="error")]}

    def _should_continue(self, state: AutonomousState):
        """Determine whether to continue with tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the last message has tool calls, continue to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        
        # Otherwise, end the conversation
        return "end"

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Any:
        """Execute a specific tool call."""
        try:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Find the tool in our tool manager
            if tool_name in self.tool_manager.tools:
                tool = self.tool_manager.tools[tool_name]
                return tool.func(**tool_args)
            else:
                return f"Tool '{tool_name}' not found"
                
        except Exception as e:
            return f"Error executing tool: {e}"

    def process_request_autonomous(self, user_input: str) -> Dict[str, Any]:
        """Process requests using autonomous ReAct pattern with self-healing."""
        start_time = time.time()
        
        try:
            # Initialize autonomous state
            initial_state = AutonomousState(
                messages=[HumanMessage(content=user_input)],
                current_task=user_input,
                task_history=[],
                error_count=0,
                success_count=0,
                learning_data={},
                self_healing_active=True,
                agent_rotation_count=0
            )
            
            # Use LangGraph workflow if available
            if hasattr(self, 'workflow') and self.workflow:
                result = self.workflow.invoke(initial_state)
                
                # Extract final response
                final_messages = result.get("messages", [])
                if final_messages:
                    last_message = final_messages[-1]
                    if hasattr(last_message, 'content'):
                        response_content = last_message.content
                    else:
                        response_content = str(last_message)
                else:
                    response_content = "Task completed successfully"
                
                duration = time.time() - start_time
                
                # Learn from this interaction
                self._learn_from_interaction(user_input, response_content, True, duration)
                
                return {
                    "success": True,
                    "result": response_content,
                    "duration": duration,
                    "method": "autonomous_react"
                }
            else:
                # Fallback to traditional processing
                return self._fallback_processing(user_input, start_time)
                
        except Exception as e:
            # Self-healing: analyze error and attempt recovery
            if hasattr(self, 'self_healing') and self.self_healing:
                error_analysis = self.self_healing.analyze_error(str(e), {"input": user_input})
                self.logger.info(f"Error analysis: {error_analysis}")
                
                # Attempt recovery
                if error_analysis['frequency'] < 3:
                    return self._attempt_recovery(user_input, str(e), start_time)
                else:
                    return {
                        "success": False,
                        "error": f"Persistent error after multiple attempts: {e}",
                        "error_analysis": error_analysis
                    }
            else:
                # Simple error handling without self-healing
                self.logger.error(f"Autonomous processing failed: {e}")
                return {
                    "success": False,
                    "error": f"Task failed: {e}",
                    "duration": time.time() - start_time
                }

    def _fallback_processing(self, user_input: str, start_time: float) -> Dict[str, Any]:
        """Fallback processing when autonomous systems fail."""
        try:
            # Use the full AI processing as fallback
            self.process_request(user_input)
            duration = time.time() - start_time
            
            return {
                "success": True,
                "result": "Task processed using full AI system",
                "duration": duration,
                "method": "ai_processing_fallback"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"AI processing failed: {e}",
                "method": "fallback"
            }


    def _attempt_recovery(self, user_input: str, error: str, start_time: float) -> Dict[str, Any]:
        """Attempt to recover from errors using different strategies."""
        try:
            # Try using CrewAI agents for recovery
            if self.crew_agents:
                recovery_task = Task(
                    description=f"Recover from error: {error}. Original request: {user_input}",
                    agent=self.crew_agents["debugger"],
                    expected_output="Successful recovery or alternative solution"
                )
                
                crew = Crew(
                    agents=[self.crew_agents["debugger"]],
                    tasks=[recovery_task],
                    process=Process.sequential,
                    verbose=True
                )
                
                result = crew.kickoff()
                duration = time.time() - start_time
                
                return {
                    "success": True,
                    "result": str(result),
                    "duration": duration,
                    "method": "recovery_crewai"
                }
            else:
                # Simple retry with exponential backoff
                time.sleep(1)
                return self._fallback_processing(user_input, start_time)
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Recovery attempt failed: {e}",
                "method": "recovery_failed"
            }

    def _learn_from_interaction(self, input_text: str, output_text: str, success: bool, duration: float):
        """Learn from interactions to improve future performance."""
        try:
            interaction_data = {
                "input": input_text,
                "output": output_text,
                "success": success,
                "duration": duration,
                "timestamp": time.time()
            }
            
            if success:
                # Store successful patterns
                pattern_key = hashlib.md5(input_text.encode()).hexdigest()[:8]
                if pattern_key not in self.context["success_patterns"]:
                    self.context["success_patterns"][pattern_key] = []
                
                self.context["success_patterns"][pattern_key].append(interaction_data)
                
                # Keep only recent successful patterns
                if len(self.context["success_patterns"][pattern_key]) > 10:
                    self.context["success_patterns"][pattern_key] = \
                        self.context["success_patterns"][pattern_key][-10:]
            else:
                # Store error patterns for analysis
                error_key = hashlib.md5(input_text.encode()).hexdigest()[:8]
                if error_key not in self.context["error_patterns"]:
                    self.context["error_patterns"][error_key] = []
                
                self.context["error_patterns"][error_key].append(interaction_data)
                
                # Keep only recent error patterns
                if len(self.context["error_patterns"][error_key]) > 5:
                    self.context["error_patterns"][error_key] = \
                        self.context["error_patterns"][error_key][-5:]
            
            self.logger.debug(f"Learned from interaction: success={success}, duration={duration:.2f}s")
            
        except Exception as e:
            self.logger.warning(f"Learning from interaction failed: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        self.executor.shutdown(wait=True)

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

    def _get_running_processes_from_context(self) -> Dict[str, Any]:
        """Extract information about running processes from context history."""
        running_processes = {}
        
        # Check recent execution history for process information
        if "last_result" in self.context:
            last_result = self.context["last_result"]
            if isinstance(last_result, dict) and "full_result" in last_result:
                full_result = last_result["full_result"]
                if isinstance(full_result, dict) and "process_id" in full_result:
                    process_id = full_result["process_id"]
                    running_processes[process_id] = {
                        "process_id": process_id,
                        "pid": full_result.get("pid"),
                        "command": full_result.get("command"),
                        "status": full_result.get("status"),
                        "timestamp": last_result.get("timestamp")
                    }
        
        # Also check conversation history for process information
        if "conversation_history" in self.context:
            for conv in self.context["conversation_history"]:
                if isinstance(conv, dict) and "ai" in conv:
                    # Look for process information in AI responses
                    ai_response = conv["ai"]
                    if isinstance(ai_response, str) and "process_id" in ai_response.lower():
                        # Try to extract process information from the response
                        import re
                        process_matches = re.findall(r'process[_\s]*id[:\s]*([a-f0-9]{8})', ai_response, re.IGNORECASE)
                        for process_id in process_matches:
                            running_processes[process_id] = {
                                "process_id": process_id,
                                "source": "conversation_history"
                            }
        
        return running_processes

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


    def _set_active_session_tool(self, session_id: str) -> Dict[str, Any]:
        """Tool wrapper for setting active session."""
        try:
            success = self.context_manager.set_active_session(session_id)
            if success:
                session = self.context_manager.get_active_session()
                return {
                    "success": True,
                    "output": f"Activated session: {session_id}",
                }
            else:
                return {"success": False, "error": "Session not found"}
        except Exception as e:
            return {"success": False, "error": f"Failed to set active session: {e}"}

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
                    timestamp = getattr(entry, 'timestamp', 0)
                    time_str = (
                        datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                        if timestamp
                        else "Unknown"
                    )
                    entry_type = getattr(entry, 'context_type', 'unknown')
                    if hasattr(entry_type, 'value'):
                        entry_type = entry_type.value
                    content = getattr(entry, 'content', '')
                    formatted_results.append(
                        f"[{time_str}] {entry_type}: {content[:200]}..."
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
                    timestamp = getattr(entry, 'timestamp', 0)
                    time_str = (
                        datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
                        if timestamp
                        else "Unknown"
                    )
                    entry_type = getattr(entry, 'context_type', 'unknown')
                    if hasattr(entry_type, 'value'):
                        entry_type = entry_type.value
                    content = getattr(entry, 'content', '')
                    formatted_results.append(
                        f"[{time_str}] {entry_type}: {content[:200]}..."
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
                    timestamp = getattr(entry, 'timestamp', 0)
                    time_str = (
                        datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
                        if timestamp
                        else "Unknown"
                    )
                    entry_type = getattr(entry, 'context_type', 'unknown')
                    if hasattr(entry_type, 'value'):
                        entry_type = entry_type.value
                    content = getattr(entry, 'content', '')
                    formatted_results.append(
                        f"[{time_str}] {entry_type}: {content[:200]}..."
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
                doc="Execute shell commands asynchronously for long-running or blocking commands. Usage: run_shell_async(command, timeout=300)",
                is_dynamic=False,
                func=self.base_tools.run_shell_async,
            ),
            Tool(
                name="open_application",
                code="",
                doc="Launch GUI applications in background (gedit, firefox, code, etc). PREFERRED over run_shell for apps. Usage: open_application(app_name, args='')",
                is_dynamic=False,
                func=self.base_tools.open_application,
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
                name="set_active_session",
                code="",
                doc="Set active session. Usage: set_active_session(session_id)",
                is_dynamic=False,
                func=self._set_active_session_tool,
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
                doc="Type text at specific coordinates or current position. Usage: type_text(text, x=None, y=None, window_title=None, process_name=None)",
                is_dynamic=False,
                func=self.base_tools.type_text,
            ),
            Tool(
                name="focus_window",
                code="",
                doc="Focus on a window by title, process name, or process ID. Usage: focus_window(window_title=None, process_name=None, process_id=None)",
                is_dynamic=False,
                func=self.base_tools.focus_window,
            ),
            Tool(
                name="press_key",
                code="",
                doc="Press key combination. Usage: press_key(key)",
                is_dynamic=False,
                func=self.base_tools.press_key,
            ),
            Tool(
                name="execute_gui_actions",
                code="",
                doc="Execute a list of GUI actions (click, type, key_press). Usage: execute_gui_actions(actions)",
                is_dynamic=False,
                func=self.base_tools.execute_gui_actions,
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

        # Save all registered tools to database
        self._save_tools_to_database()
        
        self.logger.success("Core tools registered successfully")

    def _save_tools_to_database(self):
        """Save all registered tools to the database with complete metadata."""
        try:
            self.logger.info("Saving tools to database...")
            
            for tool_name, tool in self.tool_manager.tools.items():
                # Create comprehensive tool metadata
                tool_metadata = {
                    "name": tool.name,
                    "doc": tool.doc,
                    "is_dynamic": tool.is_dynamic,
                    "usage_count": getattr(tool, 'usage_count', 0),
                    "last_used": getattr(tool, 'last_used', 0),
                    "category": self._categorize_tool(tool),
                    "parameters": self._extract_tool_parameters(tool),
                    "examples": self._generate_tool_examples(tool),
                    "result_formats": self._extract_tool_result_formats(tool),
                    "registered_at": time.time()
                }
                
                # Save to database
                self.tool_manager.save_tool_metadata(tool_name, tool_metadata)
                
            self.logger.success(f"Saved {len(self.tool_manager.tools)} tools to database")
            
        except Exception as e:
            self.logger.error(f"Failed to save tools to database: {e}")

    def _categorize_tool(self, tool) -> str:
        """Categorize tools based on their functionality and assign to agents."""
        name = tool.name.lower()
        doc = tool.doc.lower()
        
        # Enhanced categorization with agent assignments
        if any(keyword in name or keyword in doc for keyword in ['list', 'change_dir', 'find_files', 'navigate', 'search_directory']):
            return "explorer_agent"
        elif any(keyword in name or keyword in doc for keyword in ['shell', 'command', 'run', 'read_file', 'write_file', 'create_directory', 'delete_file']):
            return "executor_agent"
        elif any(keyword in name or keyword in doc for keyword in ['screen', 'click', 'type', 'mouse', 'scroll', 'drag', 'press_key', 'window']):
            return "gui_agent"
        elif any(keyword in name or keyword in doc for keyword in ['search', 'google', 'web', 'analyze', 'url', 'json', 'csv', 'image']):
            return "analyzer_agent"
        elif any(keyword in name or keyword in doc for keyword in ['linter', 'process_info', 'disk_usage', 'large_files', 'directory_size', 'system_info']):
            return "debugger_agent"
        elif any(keyword in name or keyword in doc for keyword in ['voice', 'speak', 'listen']):
            return "voice_agent"
        elif any(keyword in name or keyword in doc for keyword in ['install', 'package', 'dependency']):
            return "package_agent"
        else:
            return "general_agent"

    def _extract_tool_parameters(self, tool) -> list:
        """Extract parameter information from tool documentation."""
        import re
        doc = tool.doc
        
        # Look for usage patterns
        usage_match = re.search(r'Usage:\s*(\w+)\((.*?)\)', doc)
        if usage_match:
            params_str = usage_match.group(2)
            if params_str.strip():
                # Parse parameters
                params = []
                for param in params_str.split(','):
                    param = param.strip()
                    if '=' in param:
                        name, default = param.split('=', 1)
                        params.append({
                            "name": name.strip(),
                            "default": default.strip(),
                            "required": False
                        })
                    else:
                        params.append({
                            "name": param.strip(),
                            "required": True
                        })
                return params
        
        return []

    def _extract_tool_result_formats(self, tool) -> dict:
        """Extract success and result format information from tool functions."""
        import inspect
        import ast
        
        result_formats = {
            "success_format": {},
            "error_format": {},
            "common_fields": []
        }
        
        try:
            # Get the function source code
            if hasattr(tool, 'func') and callable(tool.func):
                try:
                    source = inspect.getsource(tool.func)
                except (OSError, TypeError):
                    # Skip if source cannot be retrieved
                    return result_formats
                
                # Parse the AST to find return statements
                try:
                    tree = ast.parse(source)
                except SyntaxError:
                    # Skip if source has syntax errors
                    return result_formats
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Return) and node.value:
                        # Analyze return statements
                        if isinstance(node.value, ast.Dict):
                            # Extract keys from dictionary return
                            keys = []
                            for key_node in node.value.keys:
                                if isinstance(key_node, ast.Constant):
                                    keys.append(key_node.value)
                                elif isinstance(key_node, ast.Str):  # Python < 3.8
                                    keys.append(key_node.s)
                            
                            # Categorize based on common patterns
                            if "success" in keys:
                                if any(key in keys for key in ["error", "stderr"]):
                                    result_formats["error_format"] = {
                                        "fields": keys,
                                        "description": "Error return format"
                                    }
                                else:
                                    result_formats["success_format"] = {
                                        "fields": keys,
                                        "description": "Success return format"
                                    }
                            
                            # Track common fields
                            result_formats["common_fields"].extend(keys)
                
                # Remove duplicates from common fields
                result_formats["common_fields"] = list(set(result_formats["common_fields"]))
                
        except Exception as e:
            self.logger.debug(f"Could not extract result formats for {tool.name}: {e}")
        
        # Add specific format information for known GUI tools
        if tool.name in ["type_text", "click_screen", "press_key", "move_mouse", "drag_mouse", "focus_window"]:
            result_formats["success_format"] = {
                "fields": ["success", "output", "message", "stderr"],
                "description": "GUI tool success format with error checking"
            }
            result_formats["error_format"] = {
                "fields": ["success", "error", "output", "stderr"],
                "description": "GUI tool error format with detailed error info"
            }
            result_formats["common_fields"] = ["success", "output", "error", "message", "stderr"]
        
        return result_formats

    def _generate_tool_examples(self, tool) -> list:
        """Generate example usage for tools."""
        name = tool.name
        doc = tool.doc
        
        # Extract examples from doc or generate based on tool type
        examples = []
        
        # Look for existing examples in documentation
        import re
        example_matches = re.findall(r'Example[s]?:?\s*(.+?)(?:\n|$)', doc, re.IGNORECASE)
        examples.extend(example_matches)
        
        # Generate basic examples if none found
        if not examples:
            if name == "run_shell":
                examples = ["run_shell('ls -la')", "run_shell('python script.py')"]
            elif name == "read_file":
                examples = ["read_file('config.txt')", "read_file('/path/to/file.py')"]
            elif name == "write_file":
                examples = ["write_file('output.txt', 'Hello World')"]
            elif name == "list_dir":
                examples = ["list_dir('.')", "list_dir('/home/user')"]
            elif name == "google_search":
                examples = ["google_search('python tutorials')", "google_search('AI news', num_results=5)"]
        
        return examples

    def _generate_dynamic_tool_descriptions(self) -> str:
        """Generate comprehensive tool descriptions from database metadata."""
        try:
            # Get all tool metadata from database
            all_metadata = self.tool_manager.get_all_tool_metadata()
            
            if not all_metadata:
                # Fallback to basic tool descriptions if no metadata
                return "\n".join(
                    [f"- {tool.name}: {tool.doc}" for tool in self.tool_manager.tools.values()]
                )
            
            # Group tools by category
            categories = {}
            for tool_name, metadata in all_metadata.items():
                category = metadata.get("category", "general")
                if category not in categories:
                    categories[category] = []
                categories[category].append((tool_name, metadata))
            
            # Generate comprehensive descriptions
            descriptions = ["**AVAILABLE TOOLS:**\n"]
            
            # Category order for better organization
            category_order = [
                "system_commands", "file_operations", "gui_automation", 
                "web_search", "voice_operations", "package_management", 
                "system_info", "general"
            ]
            
            for category in category_order:
                if category in categories:
                    category_name = category.replace("_", " ").title()
                    descriptions.append(f"\n**{category_name}:**")
                    
                    for tool_name, metadata in categories[category]:
                        descriptions.append(f"\n🔧 **{tool_name}**")
                        descriptions.append(f"   📝 {metadata.get('description', self.tool_manager.tools[tool_name].doc if tool_name in self.tool_manager.tools else 'No description')}")
                        
                        # Add parameters if available
                        parameters = metadata.get("parameters", [])
                        if parameters:
                            descriptions.append("   📋 Parameters:")
                            for param in parameters:
                                param_str = f"     • {param['name']}"
                                if not param.get("required", True):
                                    param_str += f" (optional, default: {param.get('default', 'None')})"
                                else:
                                    param_str += " (required)"
                                descriptions.append(param_str)
                        
                        # Add examples if available
                        examples = metadata.get("examples", [])
                        if examples:
                            descriptions.append("   💡 Examples:")
                            for example in examples[:3]:  # Limit to 3 examples
                                descriptions.append(f"     • {example}")
                        
                        # Add usage statistics
                        usage_count = metadata.get("usage_count", 0)
                        if usage_count > 0:
                            descriptions.append(f"   📊 Used {usage_count} times")
                    
                    del categories[category]
            
            # Add any remaining categories
            for category, tools in categories.items():
                category_name = category.replace("_", " ").title()
                descriptions.append(f"\n**{category_name}:**")
                for tool_name, metadata in tools:
                    descriptions.append(f"\n🔧 **{tool_name}**")
                    descriptions.append(f"   📝 {metadata.get('description', self.tool_manager.tools[tool_name].doc if tool_name in self.tool_manager.tools else 'No description')}")
            
            return "\n".join(descriptions)
            
        except Exception as e:
            self.logger.error(f"Error generating dynamic tool descriptions: {e}")
            # Fallback to basic descriptions
            return "\n".join(
                [f"- {tool.name}: {tool.doc}" for tool in self.tool_manager.tools.values()]
            )

    def _generate_agent_tool_assignments(self) -> str:
        """Generate dynamic agent tool assignments based on tool categories."""
        try:
            # Get all tool metadata from database
            all_metadata = self.tool_manager.get_all_tool_metadata()
            
            if not all_metadata:
                return "No tools available. System is initializing..."
            
            # Group tools by agent category
            agent_tools = {}
            for tool_name, metadata in all_metadata.items():
                agent = metadata.get("category", "general_agent")
                if agent not in agent_tools:
                    agent_tools[agent] = []
                agent_tools[agent].append((tool_name, metadata))
            
            # Agent descriptions and emojis
            agent_info = {
                "explorer_agent": ("🔍 EXPLORER AGENT", "Discovery and navigation tools"),
                "executor_agent": ("🛠️ EXECUTOR AGENT", "File operations and system commands"),
                "gui_agent": ("🎯 GUI AGENT", "Screen interaction and automation tools"),
                "analyzer_agent": ("📊 ANALYZER AGENT", "Data analysis and web search tools"),
                "debugger_agent": ("🔧 DEBUGGER AGENT", "System monitoring and debugging tools"),
                "voice_agent": ("🎤 VOICE AGENT", "Speech recognition and synthesis tools"),
                "package_agent": ("📦 PACKAGE AGENT", "Package and dependency management"),
                "general_agent": ("⚙️ GENERAL AGENT", "General purpose tools")
            }
            
            # Generate agent sections
            descriptions = []
            
            # Process agents in priority order
            agent_order = ["explorer_agent", "executor_agent", "gui_agent", "analyzer_agent", "debugger_agent", "voice_agent", "package_agent", "general_agent"]
            
            for agent in agent_order:
                if agent in agent_tools:
                    agent_name, agent_desc = agent_info[agent]
                    descriptions.append(f"\n### **{agent_name}**")
                    descriptions.append(f"*{agent_desc}*")
                    
                    for tool_name, metadata in agent_tools[agent]:
                        descriptions.append(f"\n🔧 **{tool_name}**")
                        descriptions.append(f"   📝 {metadata.get('description', self.tool_manager.tools[tool_name].doc if tool_name in self.tool_manager.tools else 'No description')}")
                        
                        # Add parameters if available
                        parameters = metadata.get("parameters", [])
                        if parameters:
                            descriptions.append("   📋 Parameters:")
                            for param in parameters:
                                param_str = f"     • {param['name']}"
                                if not param.get("required", True):
                                    param_str += f" (optional, default: {param.get('default', 'None')})"
                                else:
                                    param_str += " (required)"
                                descriptions.append(param_str)
                        
                        # Add examples if available
                        examples = metadata.get("examples", [])
                        if examples:
                            descriptions.append("   💡 Examples:")
                            for example in examples[:2]:  # Limit to 2 examples per tool
                                descriptions.append(f"     • {example}")
                    
                    del agent_tools[agent]
            
            # Add any remaining agents
            for agent, tools in agent_tools.items():
                agent_name, agent_desc = agent_info.get(agent, (f"🤖 {agent.upper()}", "Specialized tools"))
                descriptions.append(f"\n### **{agent_name}**")
                descriptions.append(f"*{agent_desc}*")
                
                for tool_name, metadata in tools:
                    descriptions.append(f"\n🔧 **{tool_name}**")
                    descriptions.append(f"   📝 {metadata.get('description', self.tool_manager.tools[tool_name].doc if tool_name in self.tool_manager.tools else 'No description')}")
            
            return "\n".join(descriptions)
            
        except Exception as e:
            self.logger.error(f"Error generating agent tool assignments: {e}")
            return "Error generating agent assignments. Using fallback mode."

    def _suggest_agent_for_task(self, task_description: str) -> str:
        """Suggest the most appropriate agent for a given task."""
        task_lower = task_description.lower()
        
        # Agent priority mapping based on task keywords
        agent_keywords = {
            "explorer_agent": ["explore", "find", "search", "navigate", "discover", "list", "directory"],
            "executor_agent": ["execute", "run", "create", "write", "delete", "install", "command"],
            "gui_agent": ["click", "type", "screen", "window", "mouse", "keyboard", "automate", "gui"],
            "analyzer_agent": ["analyze", "search", "google", "web", "data", "json", "csv", "image"],
            "debugger_agent": ["debug", "error", "check", "monitor", "system", "process", "disk"],
            "voice_agent": ["voice", "speak", "listen", "audio", "sound"],
            "package_agent": ["package", "install", "dependency", "pip", "npm"]
        }
        
        # Find the best matching agent
        best_agent = "general_agent"
        max_matches = 0
        
        for agent, keywords in agent_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in task_lower)
            if matches > max_matches:
                max_matches = matches
                best_agent = agent
        
        return best_agent

    def _should_rotate_agent(self) -> bool:
        """Determine if we should rotate to a different agent to prevent loops."""
        if not self.agent_rotation_enabled or len(self.agent_usage_history) < 3:
            return False
        
        # Check if we've used the same agent too many times recently
        recent_agents = list(self.agent_usage_history)[-3:]  # Last 3 uses
        if len(set(recent_agents)) == 1:  # All same agent
            return True
        
        return False

    def _get_alternative_agent(self, current_agent: str) -> str:
        """Get an alternative agent when rotation is needed."""
        agent_alternatives = {
            "explorer_agent": ["executor_agent", "analyzer_agent"],
            "executor_agent": ["explorer_agent", "debugger_agent"],
            "gui_agent": ["analyzer_agent", "executor_agent"],
            "analyzer_agent": ["executor_agent", "explorer_agent"],
            "debugger_agent": ["executor_agent", "explorer_agent"],
            "voice_agent": ["executor_agent", "gui_agent"],
            "package_agent": ["executor_agent", "debugger_agent"],
            "general_agent": ["explorer_agent", "executor_agent"]
        }
        
        alternatives = agent_alternatives.get(current_agent, ["explorer_agent", "executor_agent"])
        # Return the first alternative that hasn't been used recently
        for alt in alternatives:
            if alt not in list(self.agent_usage_history)[-2:]:
                return alt
        
        return alternatives[0]

    def _update_agent_usage(self, agent: str):
        """Update agent usage history."""
        self.agent_usage_history.append(agent)
        self.current_agent = agent

    def _refresh_tools_from_database(self):
        """Refresh tool metadata from database, useful when database is recreated."""
        try:
            self.logger.info("Refreshing tools from database...")
            
            # Get all tools currently registered
            registered_tools = list(self.tool_manager.tools.keys())
            
            # Check which tools need metadata refresh
            tools_needing_refresh = []
            for tool_name in registered_tools:
                metadata = self.tool_manager.get_tool_metadata(tool_name)
                if not metadata:
                    tools_needing_refresh.append(tool_name)
            
            # Re-save metadata for tools that don't have it
            if tools_needing_refresh:
                self.logger.info(f"Refreshing metadata for {len(tools_needing_refresh)} tools...")
                for tool_name in tools_needing_refresh:
                    if tool_name in self.tool_manager.tools:
                        tool = self.tool_manager.tools[tool_name]
                        tool_metadata = {
                            "name": tool.name,
                            "doc": tool.doc,
                            "is_dynamic": tool.is_dynamic,
                            "usage_count": getattr(tool, 'usage_count', 0),
                            "last_used": getattr(tool, 'last_used', 0),
                            "category": self._categorize_tool(tool),
                            "parameters": self._extract_tool_parameters(tool),
                            "examples": self._generate_tool_examples(tool),
                            "registered_at": time.time()
                        }
                        self.tool_manager.save_tool_metadata(tool_name, tool_metadata)
                
                self.logger.success(f"Refreshed metadata for {len(tools_needing_refresh)} tools")
            else:
                self.logger.info("All tools already have metadata in database")
                
        except Exception as e:
            self.logger.error(f"Error refreshing tools from database: {e}")

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

            # Use Gemini client
            import asyncio
            response = asyncio.run(self.gemini_client.generate_response(enhanced_prompt))

            if response.get("success"):
                # Add context entry
                from src.core.context_manager import ContextType, Priority
                from src.core.context_manager import ContextRelevance
                self.context_manager.add_context_entry(
                    ContextType.AI_RESPONSE,
                    response["text"],
                    Priority.NORMAL,
                    ContextRelevance.RELEVANT,
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
        tool_descriptions = self._generate_dynamic_tool_descriptions()

        # Build context string from context manager (actual context)
        context_string = ""
        if self.context_manager.current_session:
            # Get recent context entries from actual_context
            recent_context = self.context_manager.get_recent_context(
                20
            )  # Last 20 entries
            if recent_context:
                context_string += "**Recent Context (Last 20 entries):**\n"
                for entry in recent_context:
                    # Handle both dict and ContextEntry objects
                    if isinstance(entry, dict):
                        entry_type = entry.get('type', 'unknown')
                        content = entry.get('content', "")
                        timestamp = entry.get('timestamp', 0)
                    else:
                        entry_type = getattr(entry, 'context_type', 'unknown')
                        if hasattr(entry_type, 'value'):
                            entry_type = entry_type.value
                        content = getattr(entry, 'content', "")
                        timestamp = getattr(entry, 'timestamp', 0)
                    
                    if timestamp:
                        dt = datetime.fromtimestamp(timestamp)
                        time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                        time_short = dt.strftime("%H:%M:%S")
                    else:
                        time_str = "Unknown"
                        time_short = "Unknown"

                    if entry_type == "user_input":
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
            if hasattr(self.context_manager.current_session, 'context_summary') and self.context_manager.current_session.context_summary:
                context_string += f"**Context Summary:**\n{self.context_manager.current_session.context_summary}\n\n"

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

        # Add running processes context
        running_processes_context = ""
        running_processes = self._get_running_processes_from_context()
        if running_processes:
            running_processes_context = "\n**Currently Running Processes:**\n"
            for process_id, process_info in running_processes.items():
                command = process_info.get('command', 'Unknown')
                pid = process_info.get('pid', 'Unknown')
                status = process_info.get('status', 'Unknown')
                running_processes_context += f"- Process ID: {process_id} | PID: {pid} | Command: {command} | Status: {status}\n"
            running_processes_context += "\n💡 **TIP**: Use these process IDs with focus_window or interact_with_process tools!\n"

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

        system_info = json.dumps(self.context["system_info"], indent=2, default=str)
        initial_goal_text = (
            f"The user's original goal was: {self.context['initial_goal']}\n"
            if self.context.get("initial_goal")
            else ""
        )

        prompt_template = f"""
# 🤖 **ADVANCED AUTONOMOUS AI ASSISTANT SYSTEM** 🤖

You are a **state-of-the-art, production-ready AI Assistant** with comprehensive capabilities, advanced tool integration, and intelligent autonomous operation. This system represents the pinnacle of AI automation technology.

## 🚀 **ADVANCED CAPABILITIES OVERVIEW:**

### **🧠 Core Intelligence:**
- **Advanced Reasoning**: Multi-step logical reasoning with ReAct pattern
- **Self-Healing**: Automatic error detection, analysis, and recovery
- **Adaptive Learning**: Continuous improvement from interactions and feedback
- **Context Awareness**: Deep understanding of conversation history and system state
- **Multi-Modal Processing**: Text, images, files, system interactions, and more

### **🛠️ Comprehensive Tool Arsenal ({len(self.tool_manager.tools)} Tools Available):**
- **File Operations**: Complete file system management (create, read, write, copy, move, delete, search)
- **System Control**: Process management, system monitoring, package installation
- **GUI Automation**: Mouse control, keyboard input, screen interaction, window management
- **Web Integration**: Search, analysis, URL processing, content extraction
- **Development Tools**: Code analysis, linting, project creation, dependency management
- **Data Processing**: JSON, CSV, archive handling, structured data manipulation
- **Context Management**: Advanced session management, memory, analytics, optimization

### **🎯 Core Principles:**
- **DYNAMIC**: Never hardcode paths - always explore and discover what exists
- **ADAPTIVE**: Work with whatever you find, don't assume anything exists
- **SELF-HEALING**: Automatically resolve issues and learn from failures
- **EFFICIENT**: Use the most appropriate approach for each task
- **LEARNING**: Continuously improve based on experience
- **TOOL-AGNOSTIC**: Use any available tool as needed
- **PRODUCTION-READY**: Enterprise-grade reliability and performance

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
2. Use Explorer Agent tools to automatically find and navigate to common user directories
3. Use directory listing tools to see what's available at each level
4. Navigate step by step: root → username → Pictures → Screenshots
5. If a directory doesn't exist, try alternative paths
6. Use file search tools with patterns like "*.png", "*.jpg" to locate images

**Quick Start for Image/Screenshot Tasks:**
1. Use Explorer Agent tools to automatically find Pictures/Screenshots directories
2. Use directory listing tools to see what's in the current directory
3. Use file search tools with "*.png" or "*.jpg" to find image files
4. Use appropriate analysis tools to analyze any found images or capture current screen

**Key Navigation Strategy:**
1. First, use directory listing tools to see what's in the current directory
2. If you see a directory you need to explore (like "screenshots", "Pictures", "Desktop"), use navigation tools to go to it
3. Then use directory listing tools again to see what's inside that directory
4. Use file search tools with patterns like "*.png", "*.jpg", "*.jpeg", "*.gif" to find image files
5. Use appropriate analysis tools to analyze any image you find or capture current screen
6. If you need to find specific content, use search tools to search across multiple files

**Dynamic Discovery Examples:**
- For screenshots: Look in Desktop, Pictures, Downloads, or any folder with "screenshot" in the name
- For images: Use file search tools with patterns like "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp"
- For documents: Look in Documents, Desktop, or use file search tools with "*.pdf", "*.doc", "*.txt"
- For code: Look for folders with "src", "code", "project" in the name, or use file search tools with "*.py", "*.js", "*.html"

## 🚀 **MULTI-AGENT EXECUTION INSTRUCTIONS:**

When creating plans, assign specific agents to each step:

**Format for each step:**
```json
{{
  "step": "AGENT_NAME: Brief description of what this agent will do",
  "action": "tool_name",
  "args": {{}}
}}
```

**Example Agent Assignments:**
- `"step": "EXPLORER: Discover available directories and files"`
- `"step": "EXECUTOR: Remove the specified files using rm command"`
- `"step": "GUI AGENT: Capture screenshot to verify current state"`
- `"step": "ANALYZER: Process the search results and extract key information"`
- `"step": "DEBUGGER: Diagnose why the previous command failed"`

**Agent Selection Guidelines:**
- **EXPLORER**: Use for discovery, navigation, finding files/directories
- **EXECUTOR**: Use for file operations, shell commands, installations
- **GUI AGENT**: Use for visual interactions, screenshots, mouse/keyboard
- **ANALYZER**: Use for data processing, content analysis, search results
- **DEBUGGER**: Use for error resolution, system diagnostics, fixes
- **COORDINATOR**: Use for planning, monitoring, final decisions

## 🛠️ **COMPREHENSIVE TOOL ARSENAL:**

### **📁 File & Directory Operations:**
- **read_file(path)**: Read any file content with encoding detection
- **write_file(path, content)**: Create or overwrite files with proper encoding
- **list_dir(path)**: List directory contents with detailed information
- **create_directory(path)**: Create directories with proper permissions
- **copy_file(src, dest)**: Copy files with metadata preservation
- **move_file(src, dest)**: Move/rename files efficiently
- **delete_file(path)**: Safely delete files with confirmation
- **get_file_info(path)**: Get detailed file metadata and statistics
- **search_in_file(path, pattern)**: Search for patterns within files
- **replace_in_file(path, old, new)**: Replace text patterns in files
- **find_files(pattern, directory)**: Find files matching patterns
- **get_directory_size(path)**: Calculate directory size and statistics
- **find_large_files(directory, min_size)**: Identify large files for cleanup

### **💻 System Control & Management:**
- **run_shell(command, timeout)**: Execute shell commands with timeout control
- **run_shell_async(command, timeout)**: Run commands in background
- **get_system_info()**: Get comprehensive system information
- **get_process_info()**: List and monitor running processes
- **interact_with_process(pid, action)**: Control running processes
- **install_package(package)**: Install Python packages
- **install_system_package(package)**: Install system packages
- **check_system_dependency(dependency)**: Verify system dependencies
- **run_linter(file_path)**: Run code linting and analysis

### **🖱️ GUI Automation & Interaction:**
- **get_mouse_position()**: Get current mouse coordinates
- **move_mouse(x, y)**: Move mouse to specific coordinates
- **click_screen(x, y)**: Click at specific screen coordinates
- **drag_mouse(start_x, start_y, end_x, end_y)**: Drag mouse between points
- **type_text(text)**: Type text at current cursor position
- **press_key(key)**: Press specific keys (Ctrl, Alt, Enter, etc.)
- **scroll_screen(x, y)**: Scroll screen in specified direction
- **get_active_window()**: Get information about active window
- **get_all_windows()**: List all visible windows
- **bring_window_to_front(window_title)**: Focus specific window
- **focus_window(process_id)**: Focus window by process ID
- **read_screen()**: Capture and analyze current screen
- **analyze_screen_actions()**: Analyze screen for interactive elements

### **🌐 Web & Search Operations:**
- **google_search(query)**: Search Google and return results
- **google_search_news(query)**: Search Google News
- **google_search_images(query)**: Search Google Images
- **enhanced_web_search(query, max_results)**: Advanced web search with formatting
- **analyze_urls(urls)**: Analyze multiple URLs and extract information
- **check_browser_status()**: Check browser availability and status

### **📊 Data Processing & Analysis:**
- **read_json_file(path)**: Read and parse JSON files
- **write_json_file(path, data)**: Write data as JSON
- **read_csv_file(path)**: Read CSV files with proper parsing
- **write_csv_file(path, data)**: Write data as CSV
- **create_archive(files, archive_path)**: Create compressed archives
- **extract_archive(archive_path, dest)**: Extract archives
- **analyze_image(image_path, prompt)**: Analyze images with AI vision
- **generate_structured_output(prompt, schema)**: Generate structured data

### **🧠 Context & Memory Management:**
- **get_context_summary()**: Get current context summary
- **search_context_by_time(start, end)**: Search context by time range
- **get_context_by_date(date)**: Get context entries for specific date
- **get_context_by_hour(hour)**: Get context entries for specific hour
- **create_session(name)**: Create new context session
- **set_active_session(session_id)**: Set active session context

### **🔧 Development & System Tools:**
- **search_directory(directory, pattern)**: Search across directory contents
- **replace_in_multiple_files(files, old, new)**: Replace text in multiple files
- **navigate_to_user_directories()**: Navigate to common user directories
- **get_system_disk_usage()**: Get disk usage statistics

### **📈 Advanced Features:**
- **complete_task(message)**: Mark task as completed with summary
- **open_application(app_name)**: Launch applications
- **create_and_save_tool(name, code)**: Create custom tools dynamically
- **execute_gui_actions(actions)**: Execute complex GUI action sequences

**Total Available Tools: {len(self.tool_manager.tools)}**

{tool_descriptions}

**System Information:**
{system_info}

**Conversation & Memory Context:**
{context_string}

**Execution History (to prevent repeated failures):**
{execution_history_string}

{last_result_context}

{running_processes_context}

{last_error_context}

## 🎯 **MULTI-AGENT EXECUTION GUIDELINES:**

### **COORDINATOR AGENT Responsibilities:**
- Analyze user requests and create comprehensive plans
- Assign specific agents to each step based on task requirements
- Monitor overall progress and quality
- Make final decisions on task completion
- Coordinate between agents when handoffs are needed

### **AGENT COLLABORATION RULES:**
- **Seamless Handoffs**: Each agent builds on previous agent's work
- **Context Sharing**: Agents share relevant information with each other
- **Error Escalation**: When one agent fails, others step in to help
- **Quality Verification**: Multiple agents verify results before completion
- **Adaptive Planning**: Plans adjust based on agent discoveries

### **EFFICIENT EXECUTION:**
- Use the right agent for the right task
- Avoid repetitive commands and actions
- Learn from previous failures and successes
- Complete the full intent, not just the first step
- Verify results by checking actual outcomes

### **DYNAMIC APPROACH:**
- Never hardcode paths or assumptions
- Explore and discover everything dynamically
- Adapt to whatever you find in the system
- Use context from previous steps to inform next actions
- Try different approaches when something fails

## 🎯 **CURRENT TASK:**
{initial_goal_text}

## 🧠 **ADVANCED AI ASSISTANT INSTRUCTIONS:**

**COORDINATOR AGENT**: As an advanced AI assistant with comprehensive capabilities, analyze the user's request and create a detailed multi-agent execution plan. Your role is to:

1. **Understand the Complete Intent**: Analyze what the user truly wants to accomplish
2. **Leverage Your Tool Arsenal**: Use the most appropriate tools from your comprehensive collection
3. **Create Intelligent Plans**: Design efficient, multi-step approaches that solve the complete problem
4. **Assign Specialized Agents**: Deploy the right agent for each specific task type
5. **Ensure Quality Results**: Verify outcomes and provide comprehensive solutions

**Your Advanced Capabilities Include:**
- **65+ Specialized Tools** for every possible task type
- **Multi-Agent Coordination** for complex workflows
- **Self-Healing Error Recovery** for robust operation
- **Context-Aware Processing** for intelligent decision making
- **Production-Grade Reliability** for enterprise-level tasks

**Remember to be dynamic, adaptive, and efficient in your approach.**

**CRITICAL REQUIREMENTS:**
1. **MUST use agent assignments**: Every step must start with "AGENT_NAME: description"
2. **MUST be efficient**: No repetitive commands or unnecessary steps
3. **MUST complete tasks**: Use complete_task tool when finished
4. **MUST be dynamic**: Adapt to what you find, don't assume anything exists
5. **MUST prevent infinite loops**: Only avoid truly repetitive patterns (same command 3+ times in a row)
6. **MUST show results**: Display actual content from searches, file reads, and analysis
7. **MUST handle multi-step tasks**: If task has "and", "then", or multiple actions, execute ALL steps
8. **MUST handle complex projects**: For project creation, create all necessary files and directories systematically
9. **MUST use proper project structure**: Organize files in logical directories (templates/, static/, models/, etc.)

**Example proper format:**
```json
{{
  "plan": [
    {{
      "step": "EXPLORER: Discover current directory contents",
      "action": "list_dir",
      "args": {{"path": "."}}
    }},
    {{
      "step": "COORDINATOR: Complete the listing task",
      "action": "complete_task", 
      "args": {{"message": "Successfully listed directory contents"}}
    }}
  ]
}}
```

**ASYNC PROCESS MANAGEMENT:**
- `run_shell_async(command, timeout=0)` - Start process in background, returns process_id immediately
- `interact_with_process(process_id, "status")` - Check if process is still running
- `interact_with_process(process_id, "get_output")` - Get real-time output from process

**GUI AUTOMATION EXECUTION:**
- Use GUI Agent tools to analyze screen AND automatically execute actions
- Execute GUI actions directly when coordinates are known
- For GUI tasks: Use screen analysis tools first - they will both analyze and execute automatically
- The system now actually performs clicks, typing, and key presses instead of just analyzing
- Use process interaction tools to send input to running processes
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
            from src.core.context_manager import ContextType, Priority
            from src.core.context_manager import ContextRelevance
            self.context_manager.add_context_entry(ContextType.USER_INPUT, user_input, Priority.NORMAL, ContextRelevance.RELEVANT)

            # Reset execution history for new task
            self.context["execution_history"].clear()

            # Record request start
            metrics_collector.increment_counter("requests_started")

            self.logger.debug(f"Processing Request: {user_input}")

            max_retries = 5  # Limit retries to prevent infinite loops
            retry_count = 0
            
            while retry_count < max_retries:
                retry_count += 1
                self.logger.debug(f"Attempt {retry_count}/{max_retries}")
                
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
                execution_result = self._execute_plan(plan_data)
                if execution_result:
                    success = True
                    break
                else:
                    # Check if we've reached max retries
                    if retry_count >= max_retries:
                        self.logger.error(f"Max retries ({max_retries}) reached. Stopping execution.")
                        self.context["status"] = f"Task failed after {max_retries} attempts. Please provide more specific instructions or try a different approach."
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

    async def process_request_async(self, user_input: str) -> Dict[str, Any]:
        """Process requests asynchronously with better performance."""
        start_time = time.time()

        try:
            # Run multiple tasks concurrently
            tasks = [
                self._analyze_request_async(user_input),
                self._check_context_async(),
                self._prepare_tools_async(),
                self._validate_input_async(user_input),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            analysis_result = (
                results[0] if not isinstance(results[0], Exception) else None
            )
            context_result = (
                results[1] if not isinstance(results[1], Exception) else None
            )
            tools_result = results[2] if not isinstance(results[2], Exception) else None
            validation_result = (
                results[3] if not isinstance(results[3], Exception) else None
            )

            # Validate input
            if not validation_result:
                return {"success": False, "error": "Input validation failed"}

            # Generate plan asynchronously
            plan = await self._generate_plan_async(
                user_input, analysis_result, context_result
            )

            if not plan:
                return {"success": False, "error": "Failed to generate plan"}

            # Execute plan asynchronously
            execution_result = await self._execute_plan_async(plan)

            duration = time.time() - start_time
            self.logger.info(f"Request processed in {duration:.2f} seconds")

            return {
                "success": True,
                "result": execution_result,
                "duration": duration,
                "plan": plan,
            }

        except Exception as e:
            self.logger.error(f"Error in async request processing: {e}")
            return {"success": False, "error": str(e)}

    async def _analyze_request_async(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Analyze request asynchronously."""
        try:
            # Run CPU-intensive analysis in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, self._analyze_request_sync, user_input
            )
            return result
        except Exception as e:
            self.logger.error(f"Error in async request analysis: {e}")
            return None

    def _analyze_request_sync(self, user_input: str) -> Dict[str, Any]:
        """Synchronous request analysis."""
        # Simple analysis logic - can be enhanced
        return {
            "intent": "general",
            "complexity": len(user_input.split()),
            "requires_tools": True,
        }

    async def _check_context_async(self) -> Optional[Dict[str, Any]]:
        """Check context asynchronously."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, self.context_manager.get_context_summary
            )
            return result
        except Exception as e:
            self.logger.error(f"Error in async context check: {e}")
            return None

    async def _prepare_tools_async(self) -> Optional[Dict[str, Any]]:
        """Prepare tools asynchronously."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                lambda: {"available_tools": list(self.tool_manager.tools.keys())},
            )
            return result
        except Exception as e:
            self.logger.error(f"Error in async tool preparation: {e}")
            return None

    async def _validate_input_async(self, user_input: str) -> bool:
        """Validate input asynchronously."""
        try:
            # Basic validation
            if not user_input or not user_input.strip():
                return False

            # Check for dangerous patterns
            dangerous_patterns = ["<script", "javascript:", "eval("]
            for pattern in dangerous_patterns:
                if pattern in user_input.lower():
                    return False

            return True
        except Exception as e:
            self.logger.error(f"Error in async input validation: {e}")
            return False

    async def _generate_plan_async(
        self, user_input: str, analysis: Optional[Dict], context: Optional[Dict]
    ) -> Optional[Dict[str, Any]]:
        """Generate plan asynchronously."""
        try:
            # Create prompt
            prompt = self._construct_prompt()

            # Make async API call
            response = await self._make_async_api_call(prompt)

            if response:
                return self._parse_response(response)

            return None
        except Exception as e:
            self.logger.error(f"Error in async plan generation: {e}")
            return None

    async def _make_async_api_call(self, prompt: str) -> Optional[str]:
        """Make async API call to Gemini."""
        try:
            async with self.semaphore:  # Limit concurrent API calls
                if not self.session:
                    self.session = aiohttp.ClientSession()

                headers = {
                    "Content-Type": "application/json",
                }

                data = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 8192,
                    },
                }

                url = f"{self.gemini_client.base_url}/models/{self.gemini_client.api_key}:generateContent?key={self.gemini_client.current_api_key}"

                async with self.session.post(
                    url, headers=headers, json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return (
                            result.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                        )
                    else:
                        self.logger.error(f"API call failed with status {response.status}")
                        return None

        except Exception as e:
            self.logger.error(f"Error in async API call: {e}")
            return None

    async def _execute_plan_async(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute plan asynchronously."""
        try:
            steps = plan.get("plan", [])
            results = []

            # Execute steps concurrently where possible
            for step in steps:
                result = await self._execute_step_async(step)
                results.append(result)

                # If step failed, stop execution
                if not result.get("success", False):
                    break

            return {
                "success": all(r.get("success", False) for r in results),
                "results": results,
                "total_steps": len(steps),
            }

        except Exception as e:
            self.logger.error(f"Error in async plan execution: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_step_async(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step asynchronously."""
        try:
            action = step.get("action")
            args = step.get("args", {})

            if action not in self.tool_manager.tools:
                return {"success": False, "error": f"Tool '{action}' not found"}

            tool = self.tool_manager.tools[action]

            # Run tool execution in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, lambda: tool.func(**args)
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in async step execution: {e}")
            return {"success": False, "error": str(e)}

    async def run_async(self):
        """Run the async AI system."""
        try:
            async with self:
                self.logger.info("Async AI System started")

                while self.active:
                    try:
                        user_input = await self._get_user_input_async()
                        if user_input.lower() in ["exit", "quit", "stop"]:
                            break

                        result = await self.process_request_async(user_input)
                        await self._display_result_async(result)

                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        self.logger.error(f"Error in async main loop: {e}")

        except Exception as e:
            self.logger.error(f"Critical error in async system: {e}")
        finally:
            self.logger.info("Async AI System stopped")

    async def _get_user_input_async(self) -> str:
        """Get user input asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, input, "> ")

    async def _display_result_async(self, result: Dict[str, Any]):
        """Display result asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self._display_result_sync, result)

    def _display_result_sync(self, result: Dict[str, Any]):
        """Synchronous result display."""
        if result.get("success"):
            print(f"✅ Task completed successfully")
            if "duration" in result:
                print(f"⏱️  Duration: {result['duration']:.2f} seconds")
        else:
            print(f"❌ Task failed: {result.get('error', 'Unknown error')}")

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
                # Check for retry patterns but allow legitimate actions
                if ("again" in step_desc or "retry" in step_desc) and "previous" in step_desc:
                    # Allow if the step is substantial (more than just retry language)
                    # This prevents blocking legitimate requests while still catching pure retry loops
                    is_substantive = len(step_desc.split()) > 4
                    
                    if not is_substantive:
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
                
                # No limits on window management - system should be able to check windows as needed
                # Only check for truly excessive patterns (infinite loops)
                if tool_name == "get_all_windows":
                    window_calls = sum(1 for step in plan_data["plan"][:i+1] if step.get("action") == "get_all_windows")
                    if window_calls > 20:
                        return {
                            "valid": False,
                            "error": f"Step {i+1} - potential infinite loop with get_all_windows calls ({window_calls})",
                        }
                
                # Only check for truly repetitive patterns (same exact command 3 times in a row)
                if i >= 2:
                    prev_step = plan_data["plan"][i - 1]
                    prev_prev_step = plan_data["plan"][i - 2]
                    if (prev_step.get("action") == tool_name and 
                        prev_prev_step.get("action") == tool_name and
                        prev_step.get("args") == step.get("args") and
                        prev_prev_step.get("args") == step.get("args")):
                        return {
                            "valid": False,
                            "error": f"Step {i+1} repeats the same command 3 times in a row - try a different approach",
                        }
                
                # No limits on tool usage - system should be able to handle any complexity
                # Only check for truly problematic patterns like infinite loops
                action_count = sum(1 for s in plan_data["plan"][:i+1] if s.get("action") == tool_name)
                # Only block if there are more than 50 calls to the same tool (infinite loop detection)
                if action_count > 50:
                    return {
                        "valid": False,
                        "error": f"Step {i+1} - potential infinite loop detected with {tool_name} ({action_count} calls)",
                    }
                
                # Check for proper agent format (relaxed validation for dynamic agents)
                step_desc = step.get("step", "")
                # Allow any format that contains agent-like keywords or is descriptive
                agent_keywords = ["agent", "coordinator", "explorer", "executor", "gui", "debugger", "analyzer", "system", "tool"]
                if not any(keyword in step_desc.lower() for keyword in agent_keywords) and len(step_desc.split()) < 3:
                    return {
                        "valid": False,
                        "error": f"Step {i+1} should have a descriptive format with agent context",
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
            # Log step to execution history for clean display
            self.context["execution_history"].append({
                "action": action,
                "comment": comment,
                "timestamp": time.time()
            })

            # Store plan in context
            self.context["last_plan"] = {
                "action": action,
                "args": args,
                "comment": comment,
                "timestamp": time.time(),
            }

            # Check for repeated failed steps
            step_identifier = json.dumps(step_data, sort_keys=True, default=str)
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
                    self.logger.debug(f"📋 Arguments: {json.dumps(args, indent=2, default=str)}")

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
                f"Executed '{action}' with args: {json.dumps(args, indent=2, default=str)}"
            )
            if result.get("success"):
                tool_execution_content += f"\nResult: {output_text[:500]}{'...' if len(output_text) > 500 else ''}"
            else:
                tool_execution_content += (
                    f"\nError: {result.get('error', 'Unknown error')}"
                )

            from src.core.context_manager import ContextType, Priority, ContextRelevance
            self.context_manager.add_context_entry(
                ContextType.TOOL_EXECUTION,
                tool_execution_content,
                Priority.NORMAL,
                ContextRelevance.RELEVANT,
                metadata={
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

            # Auto-completion logic for simple tasks
            if result.get("success"):
                initial_goal = self.context.get("initial_goal", "").lower()
                
                # Check if the task is a simple listing task and has been completed
                if action == "list_dir" and any(keyword in initial_goal for keyword in ["list", "files", "directory", "contents", "show"]):
                    if "then" not in initial_goal and "and" not in initial_goal:
                        self.base_tools.complete_task("Successfully listed directory contents.")
                        return True
                
                # Auto-completion for get_mouse_position
                if action == "get_mouse_position" and "mouse position" in initial_goal:
                    self.base_tools.complete_task("Successfully retrieved mouse position.")
                    return True
                
                # Auto-completion for type_text
                if action == "type_text" and "type" in initial_goal and "at coordinates" in initial_goal:
                    self.base_tools.complete_task(f"Successfully typed text: {args.get('text', '')}")
                    return True
                
                # Auto-completion for get_system_info
                if action == "get_system_info" and "system info" in initial_goal:
                    self.base_tools.complete_task("Successfully retrieved system information.")
                    return True

            if output_text == "TASK_COMPLETED_SIGNAL":
                self.logger.success("Task completed successfully!")
                self.context["status"] = "Task completed successfully."
                return True

            if result.get("success"):
                # Log success to execution history for clean display
                self.logger.success(f"Tool '{action}' completed successfully")
                
                # Log the result for debugging and context
                self.logger.debug(f"Tool '{action}' result: {output_text[:200]}...")

                # Log the full output for debugging and context
                self.logger.debug(f"Step completed. Full output:\n{output_text}")
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
                from src.core.context_manager import ContextType, Priority, ContextRelevance
                error_content = f"Tool '{action}' failed with args: {json.dumps(args, indent=2, default=str)}\nError: {error_msg}"
                self.context_manager.add_context_entry(
                    ContextType.ERROR,
                    error_content,
                    Priority.HIGH,
                    ContextRelevance.IMPORTANT,
                    metadata={"action": action, "args": args, "error": error_msg},
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
        """Run the autonomous AI system with clean console output."""
        try:
            self.logger.info("Autonomous AI System started")
            if self.voice_mode:
                self.voice_tools.speak(
                    "Hello, I am an autonomous AI system with self-healing capabilities. How can I help you today?"
                )
            else:
                print("🤖 Autonomous AI System Started")
                print("Type 'exit', 'quit', or 'stop' to end the session")
                print("=" * 50)

            while self.active:
                try:
                    if self.voice_mode:
                        listen_result = self.voice_tools.listen()
                        if not listen_result["success"]:
                            self.logger.error(f"Voice input failed: {listen_result['error']}")
                            user_input = input("\n> ")
                        else:
                            user_input = listen_result["output"]
                    else:
                        user_input = input("\n> ").strip()

                    if not user_input:
                        continue

                    if user_input.lower() in ["exit", "quit", "stop", "goodbye"]:
                        if self.voice_mode:
                            self.voice_tools.speak("Goodbye!")
                        else:
                            print("👋 Goodbye!")
                        self.active = False
                        continue

                    # Show user input
                    print(f"\nYou: {user_input}")
                    
                    # Process the request using full AI system with enhanced prompt
                    self.process_request(user_input)
                    
                    # Show clean response
                    self._display_clean_response()
                    
                except KeyboardInterrupt:
                    if self.voice_mode:
                        self.voice_tools.speak("Goodbye!")
                    else:
                        print("\n👋 Goodbye!")
                    self.active = False
                    break
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    if not self.voice_mode:
                        print(f"❌ Error: {e}")
                        
        except KeyboardInterrupt:
            if self.voice_mode:
                self.voice_tools.speak("Goodbye!")
            else:
                print("\n👋 Goodbye!")
            self.active = False
        except Exception as e:
            self.logger.error(f"Unexpected error in main loop: {e}")
            if not self.voice_mode:
                print(f"❌ Error: {e}")
        finally:
            self.shutdown()

    def _display_clean_response(self):
        """Display clean response with natural conversation flow."""
        try:
            # Get the AI response from context
            ai_response = ""
            tool_executions = []
            
            # Extract AI response from conversation history
            if self.context.get("conversation_history"):
                last_conv = self.context["conversation_history"][-1]
                ai_response = last_conv.get("ai", "")
            
            # Extract tool executions from execution history
            if self.context.get("execution_history"):
                for entry in self.context["execution_history"]:
                    if entry.get("action") and entry.get("result"):
                        result = entry["result"]
                        if isinstance(result, dict) and result.get("success"):
                            output = result.get("output", "")
                            if output and len(output.strip()) > 0:
                                # Only show meaningful outputs
                                if not any(skip in output.lower() for skip in [
                                    "successfully", "completed successfully", "task completed",
                                    "command completed", "tool completed", "operation completed"
                                ]):
                                    tool_executions.append({
                                        "action": entry["action"],
                                        "args": entry.get("args", {}),
                                        "output": output
                                    })
            
            # Parse AI response to extract natural message
            natural_message = self._extract_natural_message(ai_response)
            
            # Display natural conversation flow
            if tool_executions:
                # Show tool executions in natural language
                for i, execution in enumerate(tool_executions):
                    action = execution["action"]
                    args = execution["args"]
                    output = execution["output"]
                    
                    # Convert action to natural language
                    action_desc = self._get_natural_action_description(action, args)
                    print(f"\n🔧 {action_desc}")
                    
                    # Show output in natural format
                    self._display_natural_output(action, output)
                    
                    # Add natural AI commentary between tools if needed
                    if i < len(tool_executions) - 1:  # Not the last tool
                        commentary = self._get_tool_commentary(action, output)
                        if commentary:
                            print(f"\n💭 {commentary}")
                
                # Show final AI response if available
                if natural_message and not any(word in natural_message.lower() for word in [
                    "plan", "step", "execute", "tool", "action", "args"
                ]):
                    print(f"\n💬 {natural_message}")
            else:
                # No tool executions, just show AI response
                if natural_message:
                    print(f"\n💬 {natural_message}")
                    
        except Exception as e:
            self.logger.error(f"Error displaying clean response: {e}")
            print(f"\n❌ Error displaying response: {e}")

    def _get_natural_action_description(self, action: str, args: dict) -> str:
        """Convert tool action to natural language description."""
        action_descriptions = {
            "list_dir": f"Looking at the contents of {args.get('directory', 'the directory')}",
            "google_search": f"Searching the web for '{args.get('query', 'information')}'",
            "get_system_info": "Checking system information",
            "read_file": f"Reading the file {args.get('file_path', '')}",
            "write_file": f"Creating/updating the file {args.get('file_path', '')}",
            "run_shell": f"Running command: {args.get('command', '')}",
            "create_directory": f"Creating directory {args.get('directory_path', '')}",
            "move_mouse": f"Moving mouse to position ({args.get('x', 0)}, {args.get('y', 0)})",
            "type_text": f"Typing: '{args.get('text', '')}'",
            "click_screen": f"Clicking at position ({args.get('x', 0)}, {args.get('y', 0)})",
            "get_mouse_position": "Checking current mouse position",
            "get_active_window": "Checking the active window",
            "analyze_screen_actions": "Analyzing what's on the screen",
            "search_in_file": f"Searching in file {args.get('file_path', '')} for '{args.get('pattern', '')}'",
            "find_files": f"Looking for files matching '{args.get('pattern', '')}'",
            "get_file_info": f"Getting information about {args.get('file_path', '')}",
            "copy_file": f"Copying {args.get('source', '')} to {args.get('destination', '')}",
            "move_file": f"Moving {args.get('source', '')} to {args.get('destination', '')}",
            "delete_file": f"Deleting {args.get('file_path', '')}",
            "change_dir": f"Changing to directory {args.get('directory', '')}",
            "navigate_to_user_directories": "Navigating to user directories",
            "complete_task": "Completing the task",
            "analyze_image": f"Analyzing image {args.get('image_path', '')}",
            "read_screen": "Taking a screenshot and analyzing it",
            "scroll_screen": f"Scrolling screen by ({args.get('x', 0)}, {args.get('y', 0)})",
            "press_key": f"Pressing key: {args.get('key', '')}",
            "focus_window": f"Focusing window: {args.get('window_title', '')}",
            "bring_window_to_front": f"Bringing window to front: {args.get('window_title', '')}",
            "check_browser_status": "Checking browser status",
            "install_system_package": f"Installing system package: {args.get('package_name', '')}",
            "check_system_dependency": f"Checking system dependency: {args.get('dependency', '')}",
            "execute_gui_actions": "Executing GUI actions",
            "generate_structured_output": "Generating structured output",
            "create_session": "Creating new session",
            "replace_in_file": f"Replacing text in file {args.get('file_path', '')}",
            "search_directory": f"Searching directory {args.get('directory', '')} for '{args.get('pattern', '')}'",
            "create_archive": f"Creating archive {args.get('archive_path', '')}",
            "extract_archive": f"Extracting archive {args.get('archive_path', '')}",
            "read_json_file": f"Reading JSON file {args.get('file_path', '')}",
            "write_json_file": f"Writing JSON file {args.get('file_path', '')}",
            "read_csv_file": f"Reading CSV file {args.get('file_path', '')}",
            "write_csv_file": f"Writing CSV file {args.get('file_path', '')}",
            "run_linter": f"Running linter on {args.get('file_path', '')}",
            "replace_in_multiple_files": "Replacing text in multiple files",
            "enhanced_web_search": f"Enhanced web search for '{args.get('query', '')}'",
            "analyze_urls": f"Analyzing URLs: {args.get('urls', [])}",
            "get_directory_size": f"Getting size of directory {args.get('directory', '')}",
            "find_large_files": f"Finding large files in {args.get('directory', '')}",
            "get_system_disk_usage": "Checking disk usage",
            "get_process_info": f"Getting process information for {args.get('process_name', '')}",
            "open_application": f"Opening application: {args.get('application_name', '')}",
            "interact_with_process": f"Interacting with process: {args.get('process_name', '')}",
            "install_package": f"Installing package: {args.get('package_name', '')}",
            "create_and_save_tool": f"Creating and saving tool: {args.get('tool_name', '')}",
            "set_active_session": f"Setting active session: {args.get('session_id', '')}",
            "get_context_summary": "Getting context summary",
            "search_context_by_time": f"Searching context by time: {args.get('time_range', '')}",
            "get_context_by_date": f"Getting context by date: {args.get('date', '')}",
            "get_context_by_hour": f"Getting context by hour: {args.get('hour', '')}",
        }
        
        return action_descriptions.get(action, f"Executing {action}")

    def _display_natural_output(self, action: str, output: str):
        """Display tool output in a natural, readable format."""
        if not output or len(output.strip()) == 0:
            return
            
        # Format output based on action type
        if action == "list_dir":
            print("📁 Here's what I found:")
            lines = output.split('\n')
            for line in lines[:15]:  # Limit to first 15 lines
                if line.strip():
                    print(f"   {line}")
            if len(lines) > 15:
                print(f"   ... and {len(lines) - 15} more items")
        
        elif action == "google_search":
            print("🌐 Here are the search results:")
            lines = output.split('\n')
            for line in lines[:8]:  # Limit to first 8 lines
                if line.strip():
                    print(f"   {line}")
            if len(lines) > 8:
                print(f"   ... and {len(lines) - 8} more results")
        
        elif action == "get_system_info":
            print("💻 System details:")
            for line in output.split('\n'):
                if line.strip():
                    print(f"   {line}")
        
        elif action == "read_file":
            print("📄 File contents:")
            lines = output.split('\n')
            for line in lines[:10]:  # Limit to first 10 lines
                if line.strip():
                    print(f"   {line}")
            if len(lines) > 10:
                print(f"   ... and {len(lines) - 10} more lines")
        
        elif action == "run_shell":
            print("💻 Command output:")
            lines = output.split('\n')
            for line in lines[:10]:  # Limit to first 10 lines
                if line.strip():
                    print(f"   {line}")
            if len(lines) > 10:
                print(f"   ... and {len(lines) - 10} more lines")
        
        else:
            # Generic output display
            lines = output.split('\n')
            for line in lines[:8]:  # Limit to first 8 lines
                if line.strip():
                    print(f"   {line}")
            if len(lines) > 8:
                print(f"   ... and {len(lines) - 8} more lines")

    def _get_tool_commentary(self, action: str, output: str) -> str:
        """Generate natural commentary between tool executions."""
        if action == "list_dir" and "not found" in output.lower():
            return "The directory wasn't found, let me try a different approach..."
        elif action == "google_search" and len(output.split('\n')) > 5:
            return "Found some good results! Let me process this information..."
        elif action == "read_file" and len(output) > 100:
            return "That's a substantial file. Let me analyze its contents..."
        elif action == "run_shell" and "error" in output.lower():
            return "There was an issue with that command. Let me try something else..."
        return ""

    def _extract_natural_message(self, ai_response: str) -> str:
        """Extract natural message from AI response, handling JSON format."""
        try:
            if not ai_response:
                return ""
            
            # Try to parse as JSON first
            if ai_response.strip().startswith('```json'):
                # Extract JSON from markdown
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = ai_response[json_start:json_end]
                    try:
                        response_data = json.loads(json_str)
                        plan = response_data.get("plan", [])
                        if plan and len(plan) > 0:
                            # Extract the message from the first step
                            first_step = plan[0]
                            args = first_step.get("args", {})
                            message = args.get("message", "")
                            if message:
                                return message
                    except json.JSONDecodeError:
                        pass
            
            # If not JSON or parsing failed, return the response as-is
            # but clean up any markdown formatting
            cleaned = ai_response.replace('```json', '').replace('```', '').strip()
            if cleaned.startswith('{') and cleaned.endswith('}'):
                # It's still JSON, try to extract message
                try:
                    response_data = json.loads(cleaned)
                    plan = response_data.get("plan", [])
                    if plan and len(plan) > 0:
                        first_step = plan[0]
                        args = first_step.get("args", {})
                        message = args.get("message", "")
                        if message:
                            return message
                except json.JSONDecodeError:
                    pass
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Error extracting natural message: {e}")
            return ai_response

    def shutdown(self):
        """Shutdown the AI system and clean up resources."""
        self.logger.info("Shutting down AI system...")
        self.active = False

        # Stop health server if it exists
        if hasattr(self, 'health_server') and self.health_server:
            self.health_server.stop()
            self.logger.info("Health server stopped")
        
        # Stop background task queue
        shutdown_task_queue()
        self.logger.info("Task queue stopped")

        self.memory_manager.close()
        self.tool_manager.close()
        self.execution_history.close()
        # Cleanup old sessions if needed
        if hasattr(self.context_manager, 'cleanup_old_sessions'):
            self.context_manager.cleanup_old_sessions()
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
        
        # Create error signature for tracking
        error_signature = hashlib.md5(f"{tool_name}:{error_msg}".encode()).hexdigest()[:12]
        
        # Track this error
        self.recent_errors.append({
            "signature": error_signature,
            "tool": tool_name,
            "args": args,
            "error": error_msg,
            "timestamp": time.time()
        })
        
        # Increment retry count for this error
        self.error_retry_counts[error_signature] = self.error_retry_counts.get(error_signature, 0) + 1
        
        # Detect error loops - if same error repeated 3+ times in last 5 attempts
        recent_signatures = [e["signature"] for e in list(self.recent_errors)[-5:]]
        same_error_count = recent_signatures.count(error_signature)
        
        if same_error_count >= 3:
            self.logger.error(f"🔁 ERROR LOOP DETECTED: Same error repeated {same_error_count} times!")
            self.logger.warning(f"🚫 Stopping repetitive attempts for: {tool_name}")
            self.logger.info("💡 Suggestion: This approach isn't working. Try a different method or skip this step.")
            
            # Clear this error from retry counts to prevent future attempts
            if error_signature in self.error_retry_counts:
                del self.error_retry_counts[error_signature]
            
            # Provide helpful context
            if "argument error" in error_msg or "unexpected keyword argument" in error_msg:
                self.logger.warning(f"💡 Function signature issue detected. The function may not support the arguments being passed.")
                self.logger.info(f"   Tool: {tool_name}")
                self.logger.info(f"   Args attempted: {list(args.keys())}")
            
            return False  # Don't attempt resolution for looping errors
        
        # If we've tried resolving this specific error 2+ times, don't try again
        if self.error_retry_counts.get(error_signature, 0) > 2:
            self.logger.warning(f"⚠️ Already attempted to resolve this error {self.error_retry_counts[error_signature]} times. Skipping resolution.")
            return False

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


# Convenience function for running autonomous system
async def run_autonomous_ai_system():
    """Run the autonomous AI system."""
    autonomous_system = AutonomousAISystem()
    await autonomous_system.run_async()


if __name__ == "__main__":
    # Run autonomous system by default
    asyncio.run(run_autonomous_ai_system())
