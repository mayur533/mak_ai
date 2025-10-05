"""
Async AI System for improved performance and scalability.
Provides non-blocking operations and concurrent execution capabilities.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import aiofiles

from src.core.ai_system import AISystem
from src.logging.logger import logger


class AsyncAISystem(AISystem):
    """Async version of AI system with improved performance and scalability."""

    def __init__(self, voice_mode: bool = None, max_workers: int = 4):
        """Initialize async AI system."""
        super().__init__(voice_mode)
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.session = None
        self.semaphore = asyncio.Semaphore(max_workers)

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        self.executor.shutdown(wait=True)

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
            logger.info(f"Request processed in {duration:.2f} seconds")

            return {
                "success": True,
                "result": execution_result,
                "duration": duration,
                "plan": plan,
            }

        except Exception as e:
            logger.error(f"Error in async request processing: {e}")
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
            logger.error(f"Error in async request analysis: {e}")
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
            logger.error(f"Error in async context check: {e}")
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
            logger.error(f"Error in async tool preparation: {e}")
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
            logger.error(f"Error in async input validation: {e}")
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
            logger.error(f"Error in async plan generation: {e}")
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
                        logger.error(f"API call failed with status {response.status}")
                        return None

        except Exception as e:
            logger.error(f"Error in async API call: {e}")
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
            logger.error(f"Error in async plan execution: {e}")
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
            logger.error(f"Error in async step execution: {e}")
            return {"success": False, "error": str(e)}

    async def run_async(self):
        """Run the async AI system."""
        try:
            async with self:
                logger.info("Async AI System started")

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
                        logger.error(f"Error in async main loop: {e}")

        except Exception as e:
            logger.error(f"Critical error in async system: {e}")
        finally:
            logger.info("Async AI System stopped")

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


# Convenience function for running async system
async def run_async_ai_system():
    """Run the async AI system."""
    async_system = AsyncAISystem()
    await async_system.run_async()


if __name__ == "__main__":
    asyncio.run(run_async_ai_system())
