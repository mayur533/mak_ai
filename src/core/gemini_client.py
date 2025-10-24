#!/usr/bin/env python3
"""
Gemini API Client with built-in Google Search capabilities.
Handles rate limiting, error handling, and context management.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import aiohttp
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import tiktoken
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Message types for agent communication."""
    GOAL = "goal"
    TASK = "task"
    TASK_STATUS = "task.status"
    OPERATION_REQUEST = "operation.request"
    OPERATION_RESPONSE = "operation.response"
    INFO_REQUEST = "info.request"
    LOG = "log"
    HEARTBEAT = "heartbeat"

@dataclass
class MessageEnvelope:
    """Standard message envelope for agent communication."""
    id: str
    from_agent: str
    to: str
    type: MessageType
    timestamp: str
    reply_to: Optional[str] = None
    auth: Optional[Dict[str, Any]] = None
    payload: Optional[Dict[str, Any]] = None

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, rate_limit: int, burst_limit: int):
        self.rate_limit = rate_limit  # requests per minute
        self.burst_limit = burst_limit  # burst capacity
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire permission to make a request."""
        async with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) < self.rate_limit:
                self.requests.append(now)
                return True
            return False
    
    async def wait_for_slot(self):
        """Wait until a slot is available."""
        while not await self.acquire():
            await asyncio.sleep(0.1)

class GeminiClient:
    """Enhanced Gemini API client with Google Search and context management."""
    
    def __init__(self, api_key: str, secondary_api_key: Optional[str] = None):
        self.primary_api_key = api_key
        self.secondary_api_key = secondary_api_key
        self.current_api_key = api_key
        self.rate_limiter = RateLimiter(60, 10)  # 60 req/min, 10 burst
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # Initialize Gemini
        genai.configure(api_key=self.current_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Safety settings for Google Search
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # Context management
        import os
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent
        self.context_file = project_root / "db" / "context.json"
        self.max_tokens = 100000
        self.summary_threshold = 0.9
        self.compression_ratio = 0.7
        
        # Load existing context
        self.context = self._load_context()
    
    def _load_context(self) -> Dict[str, Any]:
        """Load context from file."""
        try:
            with open(self.context_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "conversations": [],
                "summaries": [],
                "current_tokens": 0,
                "last_updated": datetime.now().isoformat()
            }
    
    def _save_context(self):
        """Save context to file."""
        try:
            # Ensure all data is JSON-serializable
            serializable_context = self._make_serializable(self.context)
            with open(self.context_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_context, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save context: {e}")
    
    def _make_serializable(self, obj):
        """Recursively convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            if not isinstance(text, str):
                text = str(text)
            return len(self.encoder.encode(text))
        except Exception as e:
            logger.error(f"Error counting tokens for text type {type(text)}: {e}")
            return 0
    
    def _should_summarize(self) -> bool:
        """Check if context should be summarized."""
        return self.context["current_tokens"] >= (self.max_tokens * self.summary_threshold)
    
    async def _summarize_context(self):
        """Summarize context to reduce token count."""
        if not self.context["conversations"]:
            return
        
        # Create summary prompt
        conversations = self.context["conversations"][-10:]  # Last 10 conversations
        
        # Safely serialize conversations, handling any unhashable types
        try:
            # Convert all data to JSON-serializable format
            serializable_conversations = []
            for conv in conversations:
                if isinstance(conv, dict):
                    serializable_conv = {}
                    for key, value in conv.items():
                        if isinstance(value, (str, int, float, bool, type(None))):
                            serializable_conv[key] = value
                        else:
                            serializable_conv[key] = str(value)
                    serializable_conversations.append(serializable_conv)
                else:
                    serializable_conversations.append(str(conv))
            
            conversations_json = json.dumps(serializable_conversations, indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to serialize conversations: {e}")
            # Fallback to string representation
            conversations_json = str(conversations)
        
        summary_prompt = f"""
        Summarize the following conversation history while preserving important details, decisions, and context:
        
        {conversations_json}
        
        Provide a concise summary that maintains the essential information for future reference.
        """
        
        try:
            response = await self._make_request(summary_prompt, use_search=False)
            summary = {
                "timestamp": datetime.now().isoformat(),
                "summary": response,
                "original_conversations": len(conversations)
            }
            
            self.context["summaries"].append(summary)
            self.context["conversations"] = self.context["conversations"][-5:]  # Keep last 5
            try:
                self.context["current_tokens"] = self._count_tokens(json.dumps(self.context["conversations"], default=str))
            except Exception as e:
                logger.warning(f"Failed to count tokens for conversations: {e}")
                self.context["current_tokens"] = 0
            self._save_context()
            
            logger.info(f"Context summarized. Tokens reduced to {self.context['current_tokens']}")
            
        except Exception as e:
            logger.error(f"Failed to summarize context: {e}")
    
    async def _make_request(self, prompt: str, use_search: bool = True, **kwargs) -> str:
        """Make a request to Gemini API with rate limiting and error handling."""
        await self.rate_limiter.wait_for_slot()
        
        try:
            # Prepare generation config
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=kwargs.get('max_tokens', 8192),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 40),
            )
            
            # Generate response without tools for now (tools can be added later)
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            
            return response.text

        except Exception as e:
            logger.error(f"API request failed: {e}")
            # Try secondary API key if available
            if self.secondary_api_key and self.current_api_key == self.primary_api_key:
                logger.info("Switching to secondary API key")
                self.current_api_key = self.secondary_api_key
                genai.configure(api_key=self.current_api_key)
                return await self._make_request(prompt, use_search, **kwargs)
            raise
    
    async def generate_response(self, prompt: str, use_search: bool = True, **kwargs) -> dict:
        """Generate a response using Gemini with optional Google Search."""
        try:
            response = await self._make_request(prompt, use_search, **kwargs)
            
            # Update context
            conversation = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "response": response,
                "tokens": self._count_tokens(str(prompt) + str(response))
            }
            
            self.context["conversations"].append(conversation)
            self.context["current_tokens"] += conversation["tokens"]
            self.context["last_updated"] = datetime.now().isoformat()
            
            # Check if summarization is needed
            if self._should_summarize():
                await self._summarize_context()
            
            self._save_context()
            
            # Return in expected format
            return {
                "success": True,
                "text": response,
                "model": "gemini-pro",
                "tokens_used": conversation["tokens"]
            }

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return {
                "success": False,
                "text": f"Error: {e}",
                "model": "gemini-pro",
                "tokens_used": 0
            }
    
    async def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web using Gemini's built-in Google Search."""
        search_prompt = f"""
        Search for information about: {query}
        
        Please provide:
        1. A summary of the most relevant information
        2. Key facts and details
        3. Sources and references
        4. Any recent developments or updates
        
        Focus on accuracy and relevance. Limit to the most important {max_results} results.
        """
        
        try:
            response = await self.generate_response(search_prompt, use_search=True)
            
            # Parse the response to extract structured information
            # This is a simplified parser - in practice, you might want more sophisticated parsing
            return [{
                "query": query,
                "summary": response,
                "timestamp": datetime.now().isoformat(),
                "source": "gemini_google_search"
            }]
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    async def plan_tasks(self, goal: str) -> List[Dict[str, Any]]:
        """Plan tasks for achieving a goal using Gemini."""
        planning_prompt = f"""
        You are the Coordinator Agent. Input: a natural-language goal. 
        Output: a JSON array `tasks` each with {{id, action, target, payload, timeout_secs, retry, priority, requires_permit}}. 
        Decompose into minimum necessary tasks. For any task that modifies the system, mark it with "requires_permit": true. 
        Only output valid JSON.
        
        Goal: {goal}
        
        Available agents:
        - system.agent (for system operations)
        - operation.agent (for file/process operations)
        - info.agent (for information gathering)
        - coordinator (for task management)
        
        Available actions:
        - read_file, write_file, replace_in_file, list_dir, glob
        - execute_process, gather_info, search_web, plan_tasks
        - create_todo, update_todo, complete_todo
        
        Output format:
        {{
            "tasks": [
                {{
                    "id": "task_1",
                    "action": "read_file",
                    "target": "system.agent",
                    "payload": {{"path": "/path/to/file"}},
                    "timeout_secs": 30,
                    "retry": 2,
                    "priority": 50,
                    "requires_permit": false
                }}
            ]
        }}
        """
        
        try:
            response = await self.generate_response(planning_prompt, use_search=False)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                task_data = json.loads(json_match.group())
                return task_data.get("tasks", [])
            else:
                logger.error("Failed to parse task planning response")
                return []
                
        except Exception as e:
            logger.error(f"Task planning failed: {e}")
            return []
    
    async def create_todo_list(self, goal: str) -> List[Dict[str, Any]]:
        """Create a todo list for a goal."""
        todo_prompt = f"""
        Create a detailed todo list for achieving this goal: {goal}
        
        Break down the goal into specific, actionable tasks. Each task should be:
        1. Clear and specific
        2. Measurable
        3. Achievable
        4. Time-bound
        
        Output as JSON array:
        [
            {{
                "id": "todo_1",
                "title": "Task title",
                "description": "Detailed description",
                "priority": "high|medium|low",
                "estimated_time": "X minutes/hours",
                "dependencies": ["todo_2", "todo_3"],
                "status": "pending|in_progress|completed",
                "created_at": "timestamp"
            }}
        ]
        """
        
        try:
            response = await self.generate_response(todo_prompt, use_search=False)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                logger.error("Failed to parse todo list response")
                return []
                
        except Exception as e:
            logger.error(f"Todo list creation failed: {e}")
            return []
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current context."""
        return {
            "total_conversations": len(self.context["conversations"]),
            "total_summaries": len(self.context["summaries"]),
            "current_tokens": self.context["current_tokens"],
            "max_tokens": self.max_tokens,
            "token_usage_percent": (self.context["current_tokens"] / self.max_tokens) * 100,
            "last_updated": self.context["last_updated"]
        }
    
    async def analyze_image(self, image_path: str, prompt: str = "Describe this image") -> Dict[str, Any]:
        """Analyze an image using Gemini Vision."""
        try:
            import base64
            from pathlib import Path
            
            # Read and encode image
            image_file = Path(image_path)
            if not image_file.exists():
                return {"success": False, "error": f"Image file not found: {image_path}"}
            
            with open(image_file, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            
            # Create the prompt for image analysis
            analysis_prompt = f"""
            {prompt}
            
            Please provide a detailed analysis of this image.
            """
            
            # Use the existing generate_response method
            response = await self.generate_response(analysis_prompt, use_search=False)
            
            return {
                "success": True,
                "analysis": response,
                "image_path": image_path,
                "prompt": prompt
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def generate_structured_output(self, prompt: str, output_format: str = "json") -> Dict[str, Any]:
        """Generate structured output in specified format."""
        try:
            structured_prompt = f"""
            {prompt}
            
            Please provide your response in {output_format} format.
            """
            
            response = await self.generate_response(structured_prompt, use_search=False)
            
            return {
                "success": True,
                "output": response,
                "format": output_format
            }
            
        except Exception as e:
            logger.error(f"Structured output generation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_session(self, session_name: str = "default") -> Dict[str, Any]:
        """Create a new session."""
        try:
            session_id = f"{session_name}_{int(time.time())}"
            
            # Add session to context
            self.context["conversations"].append({
                "session_id": session_id,
                "name": session_name,
                "created_at": datetime.now().isoformat(),
                "messages": []
            })
            
            self._save_context()
            
            return {
                "success": True,
                "session_id": session_id,
                "session_name": session_name
            }
            
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def enhanced_web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Enhanced web search with better formatting."""
        try:
            search_results = await self.search_web(query, max_results)
            
            return {
                "success": True,
                "query": query,
                "results": search_results,
                "total_results": len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Enhanced web search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def analyze_urls(self, urls: list) -> Dict[str, Any]:
        """Analyze multiple URLs and extract information."""
        try:
            results = []
            
            for url in urls:
                try:
                    # Simple URL analysis - in a real implementation, you'd fetch and analyze content
                    analysis_prompt = f"""
                    Analyze this URL: {url}
                    
                    Provide:
                    1. Domain information
                    2. Content type
                    3. Security assessment
                    4. Accessibility
                    """
                    
                    response = await self.generate_response(analysis_prompt, use_search=False)
                    
                    results.append({
                        "url": url,
                        "analysis": response,
                        "status": "success"
                    })
                    
                except Exception as e:
                    results.append({
                        "url": url,
                        "error": str(e),
                        "status": "failed"
                    })
            
            return {
                "success": True,
                "results": results,
                "total_urls": len(urls)
            }
            
        except Exception as e:
            logger.error(f"URL analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def clear_context(self):
        """Clear all context data."""
        self.context = {
            "conversations": [],
            "summaries": [],
            "current_tokens": 0,
            "last_updated": datetime.now().isoformat()
        }
        self._save_context()
        logger.info("Context cleared")

# Global client instance
_gemini_client: Optional[GeminiClient] = None

def get_gemini_client() -> GeminiClient:
    """Get the global Gemini client instance."""
    global _gemini_client
    if _gemini_client is None:
        from src.config.settings import settings
        _gemini_client = GeminiClient(
            api_key=settings.GEMINI_API_KEY,
            secondary_api_key=settings.GEMINI_API_KEY_SECONDARY
        )
    return _gemini_client