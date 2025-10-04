"""
Core AI System for the AI Assistant System.
Main orchestrator that handles AI interactions, tool management, and task execution.
"""

import json
import re
import time
from collections import deque
from typing import Dict, Any, List, Optional

import requests

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.config.settings import settings
from src.logging.logger import logger
from src.database.memory import MemoryManager, ToolManager, ExecutionHistory, Tool
from src.tools.base_tools import BaseTools
from src.tools.voice_tools import VoiceTools


class AISystem:
    """
    Core AI system that orchestrates all functionality.
    Handles AI interactions, tool management, and task execution.
    """
    
    def __init__(self, voice_mode: bool = None):
        """Initialize the AI system."""
        self.voice_mode = voice_mode if voice_mode is not None else settings.VOICE_ENABLED
        self.logger = logger
        
        # Initialize components
        self.memory_manager = MemoryManager()
        self.tool_manager = ToolManager()
        self.execution_history = ExecutionHistory()
        
        # Initialize tools
        self.base_tools = BaseTools(system=self)
        self.voice_tools = VoiceTools(system=self)
        
        # System context
        self.context = {
            "cwd": str(settings.BASE_DIR),
            "os": sys.platform,
            "python_version": sys.version,
            "system_info": {},
            "initial_goal": None,
            "pending_goal": None,
            "api_key_in_use": "primary",
            "execution_history": deque(maxlen=20),
            "conversation_history": deque(maxlen=20)
        }
        
        self.active = True
        
        # Register core tools
        self._register_core_tools()
        self._load_tools_from_files()
        self._get_initial_system_details()
        
        self.logger.success("AI System initialized successfully")
    
    def _register_core_tools(self):
        """Register core tools with the system."""
        self.logger.info("Registering core tools...")
        
        core_tools = [
            Tool(
                name="run_shell",
                code="",
                doc="Execute a shell command. Usage: run_shell(command)",
                is_dynamic=False,
                func=self.base_tools.run_shell
            ),
            Tool(
                name="run_python",
                code="",
                doc="Execute Python code. Usage: run_python(code)",
                is_dynamic=False,
                func=self.base_tools.run_python
            ),
            Tool(
                name="create_and_save_tool",
                code="",
                doc="Create and register a new tool. Usage: create_and_save_tool(tool_name, tool_code, doc_string)",
                is_dynamic=False,
                func=self.base_tools.create_and_save_tool
            ),
            Tool(
                name="install_package",
                code="",
                doc="Install a Python package. Usage: install_package(package_name)",
                is_dynamic=False,
                func=self.base_tools.install_package
            ),
            Tool(
                name="read_file",
                code="",
                doc="Read file contents. Usage: read_file(file_path)",
                is_dynamic=False,
                func=self.base_tools.read_file
            ),
            Tool(
                name="write_file",
                code="",
                doc="Write content to file. Usage: write_file(file_path, content)",
                is_dynamic=False,
                func=self.base_tools.write_file
            ),
            Tool(
                name="list_dir",
                code="",
                doc="List directory contents. Usage: list_dir(directory='.')",
                is_dynamic=False,
                func=self.base_tools.list_dir
            ),
            Tool(
                name="complete_task",
                code="",
                doc="Mark task as complete. Usage: complete_task(message)",
                is_dynamic=False,
                func=self.base_tools.complete_task
            ),
            Tool(
                name="get_system_info",
                code="",
                doc="Get system information. Usage: get_system_info()",
                is_dynamic=False,
                func=self.base_tools.get_system_info
            )
        ]
        
        # Add voice tools if enabled
        if self.voice_mode:
            voice_tools = [
                Tool(
                    name="speak",
                    code="",
                    doc="Convert text to speech. Usage: speak(text)",
                    is_dynamic=False,
                    func=self.voice_tools.speak
                ),
                Tool(
                    name="listen",
                    code="",
                    doc="Listen for speech input. Usage: listen(timeout=5)",
                    is_dynamic=False,
                    func=self.voice_tools.listen
                ),
                Tool(
                    name="test_voice_system",
                    code="",
                    doc="Test voice system functionality. Usage: test_voice_system()",
                    is_dynamic=False,
                    func=self.voice_tools.test_voice_system
                )
            ]
            core_tools.extend(voice_tools)
        
        # Register all tools
        for tool in core_tools:
            self.tool_manager.register_tool(tool)
        
        self.logger.success("Core tools registered successfully")
    
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
                        func=func
                    )
                    self.tool_manager.register_tool(tool)
                    self.logger.success(f"Tool '{tool_name}' loaded from {tool_file.name}")
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
        """Make a request to the Gemini API."""
        api_key = settings.get_api_key(self.context.get("api_key_in_use") == "secondary")
        
        if not api_key:
            self.logger.error("No API key available")
            return None
        
        headers = {'Content-Type': 'application/json'}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        params = {'key': api_key}
        
        for i in range(settings.MAX_RETRIES):
            try:
                response = requests.post(
                    settings.GEMINI_API_URL, 
                    headers=headers, 
                    json=payload, 
                    params=params, 
                    timeout=60
                )
                
                # Handle API key switching
                if response.status_code == 403 and self.context.get("api_key_in_use") == "primary":
                    if settings.GEMINI_API_KEY_SECONDARY:
                        self.logger.warning("Primary API key failed, switching to secondary")
                        self.context["api_key_in_use"] = "secondary"
                        continue
                    else:
                        self.logger.error("Primary API key failed and no secondary key available")
                        return None
                elif response.status_code == 403 and self.context.get("api_key_in_use") == "secondary":
                    self.logger.error("Both API keys failed")
                    return None
                
                response.raise_for_status()
                response_data = response.json()
                
                if 'candidates' in response_data and response_data['candidates']:
                    return response_data['candidates'][0]['content']['parts'][0]['text']
                else:
                    self.logger.error("Gemini API returned empty response")
                    return None
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed: {e}. Retrying in {2**i} seconds...", exc_info=True)
                time.sleep(2**i)
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.error(f"Failed to parse Gemini response: {e}", exc_info=True)
                return None
        
        return None
    
    def _construct_prompt(self) -> str:
        """Construct the prompt for the AI model."""
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.doc}" 
            for tool in self.tool_manager.tools.values()
        ])
        
        # Build context string
        context_string = ""
        for item in self.context["conversation_history"]:
            context_string += f"User: {item['user']}\nAI: {item['ai']}\n"
        
        # Add relevant memories
        if self.context.get("initial_goal"):
            relevant_memories = self.memory_manager.recall(self.context["initial_goal"], top_k=5)
            for item in relevant_memories:
                context_string += f"- Memory: {item.content}\n"
        
        # Add execution history
        execution_history_string = "\n".join([
            f"- Step: {item['step']}\n  Result: {item['result']}\n" 
            for item in self.context["execution_history"]
        ])
        
        system_info = json.dumps(self.context["system_info"], indent=2)
        initial_goal_text = f"The user's original goal was: {self.context['initial_goal']}\n" if self.context.get("initial_goal") else ""
        
        prompt_template = f"""
You are a highly intelligent, autonomous, and self-improving AI system. Your primary objective is to complete the user's request. You have full access to the system and a comprehensive set of tools to achieve any task.

Your core process is as follows:
1. **Analyze**: Carefully break down the user's request.
2. **Recall & Contextualize**: Use your long-term memory and recent conversation history to retrieve relevant context.
3. **Plan**: Create a detailed, step-by-step plan. Each step must be a single tool call.
4. **Execute**: Execute the plan, one step at a time.
5. **Reflect & Improve**: Analyze the output of each tool. If an error occurs, self-heal by creating a new plan to diagnose and fix the issue.

**Your Guiding Principles:**
- You have the power to create new tools using `create_and_save_tool`.
- After creating a new tool, your **NEXT** step MUST be to use `run_python` with a test case to **verify** the new tool's functionality.
- Use `run_shell` to execute system commands.
- Use `run_python` to execute Python code.
- Use `install_package` to install any necessary libraries.
- Use `read_file` and `write_file` to interact with the file system.
- Use `list_dir` to explore the file system.
- When the entire task is complete, you MUST use the `complete_task` tool with a summary message.
- **CRITICAL**: Do not repeat a failed step. If a step fails, your next plan must be to diagnose the failure and propose a new approach.

**Available Tools:**
{tool_descriptions}

**System Information:**
{system_info}

**Conversation & Memory Context:**
{context_string}

**Execution History (to prevent repeated failures):**
{execution_history_string}

{initial_goal_text}
Current Status:
{self.context.get('status', 'No specific status. Ready for a new task or step.')}

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
        """Process a user request."""
        self.context["initial_goal"] = user_input
        self.context["status"] = f"Initial user request: {user_input}"
        self.context["conversation_history"].append({"user": user_input, "ai": ""})
        
        # Reset execution history for new task
        self.context["execution_history"].clear()
        
        self.logger.banner(f"Processing Request: {user_input}")
        
        while True:
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
            if self._execute_plan(plan_data):
                break
    
    def _parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse AI response and extract plan."""
        # Try to extract JSON from markdown code block
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON from markdown: {e}")
        
        # Try to parse entire response as JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            return None
    
    def _execute_plan(self, plan_data: Dict[str, Any]) -> bool:
        """Execute a plan step."""
        plan = plan_data.get("plan")
        if not isinstance(plan, list) or not plan:
            self.logger.error("Invalid plan structure")
            return True
        
        step_data = plan[0]
        action = step_data.get("action")
        args = step_data.get("args", {})
        comment = plan_data.get("comment", "No comment provided")
        
        self.logger.step(1, 1, comment)
        self.memory_manager.remember(f"Generated plan: {comment}", {
            "type": "plan", 
            "action": action, 
            "args": args
        })
        
        # Check for repeated failed steps
        step_identifier = json.dumps(step_data, sort_keys=True)
        if self.execution_history.has_failed_before(step_identifier):
            self.logger.error("This step has previously failed. Skipping to avoid loops.")
            self.context["status"] = f"Previous plan '{comment}' has failed before. Please generate a different plan."
            return False
        
        # Execute tool
        if action not in self.tool_manager.tools:
            error_msg = f"Tool '{action}' not found"
            self.logger.error(error_msg)
            self.context["status"] = f"Plan execution failed: {error_msg}"
            self.execution_history.log_execution(step_identifier, comment, "failed", error_msg)
            return False
        
        tool = self.tool_manager.tools[action]
        
        try:
            self.tool_manager.update_tool_usage(action)
            self.logger.info(f"Executing tool: {action} with args: {args}")
            
            result = tool.func(**args)
            
            if result.get("output") == "TASK_COMPLETED_SIGNAL":
                self.logger.success("Task completed successfully!")
                self.context["status"] = "Task completed successfully."
                return True
            
            if result.get("success"):
                self.logger.success(f"Step completed. Output:\n{result.get('output')}")
                self.memory_manager.remember(
                    f"Executed '{action}' successfully. Output: {result.get('output')}", 
                    {"type": "tool_output", "tool": action}
                )
                self.context["status"] = f"Previous step '{comment}' completed successfully. What's next?"
                self.execution_history.log_execution(step_identifier, comment, "success")
            else:
                error_msg = result.get('error')
                self.logger.error(f"Step failed. Error:\n{error_msg}")
                self.memory_manager.remember(
                    f"Execution of '{action}' failed: {error_msg}", 
                    {"type": "tool_error", "tool": action}
                )
                self.context["status"] = f"Previous step failed: {error_msg}. Please generate a plan to resolve this."
                self.execution_history.log_execution(step_identifier, comment, "failed", error_msg)
            
            return False
            
        except Exception as e:
            error_msg = f"Unexpected error during execution: {e}"
            self.logger.error(error_msg, exc_info=True)
            self.memory_manager.remember(f"Unexpected error with '{action}': {e}", {"type": "tool_error"})
            self.context["status"] = f"Unexpected error during execution: {e}. Please diagnose and fix this issue."
            self.execution_history.log_execution(step_identifier, comment, "failed", error_msg)
            return False
    
    def run(self):
        """Main run loop for the AI system."""
        if self.voice_mode:
            self.voice_tools.speak("Hello, I am a self-improving AI system. How can I help you today?")
        else:
            self.logger.info("AI System ready. Type your request or 'exit' to quit.")
        
        while self.active:
            if self.voice_mode:
                listen_result = self.voice_tools.listen()
                if not listen_result["success"]:
                    self.logger.error(f"Voice input failed: {listen_result['error']}")
                    user_input = input("\n> ")
                else:
                    user_input = listen_result["output"]
            else:
                user_input = input("\n> ")
            
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit", "goodbye"]:
                if self.voice_mode:
                    self.voice_tools.speak("Goodbye!")
                else:
                    self.logger.info("Goodbye!")
                self.active = False
                continue
            
            self.process_request(user_input)
    
    def shutdown(self):
        """Shutdown the AI system and clean up resources."""
        self.logger.info("Shutting down AI system...")
        self.memory_manager.close()
        self.tool_manager.close()
        self.execution_history.close()
        self.active = False
        self.logger.success("AI system shutdown complete")
