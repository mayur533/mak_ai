"""
Enhanced Gemini API Client with advanced features.
Implements all Gemini API capabilities including function calling, context caching, and more.
"""

import json
import requests
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.config.settings import settings
from src.logging.logger import logger


@dataclass
class GeminiConfig:
    """Configuration for Gemini API requests."""
    model: str = "gemini-2.0-flash"
    temperature: float = 0.7
    max_tokens: int = 8192
    top_p: float = 0.95
    top_k: int = 40
    safety_settings: Dict[str, Any] = None
    generation_config: Dict[str, Any] = None


class GeminiClient:
    """Enhanced Gemini API client with advanced features."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        self.logger = logger
        self.api_key = settings.GEMINI_API_KEY
        self.api_key_secondary = settings.GEMINI_API_KEY_SECONDARY
        self.current_api_key = self.api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.max_retries = settings.MAX_RETRIES
        self.rate_limit_delay = 1.0
        
        # Context caching
        self.context_cache = {}
        self.cache_enabled = True
        
        # Session management
        self.current_session = None
        self.session_history = []
        
        if not self.api_key:
            self.logger.error("No Gemini API key configured")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            'Content-Type': 'application/json',
            'x-goog-api-key': self.current_api_key
        }
    
    def _switch_api_key(self):
        """Switch to secondary API key if available."""
        if self.current_api_key == self.api_key and self.api_key_secondary:
            self.current_api_key = self.api_key_secondary
            self.logger.warning("Switched to secondary API key")
        else:
            self.logger.error("No secondary API key available or already using secondary")
    
    def _reset_to_primary_api_key(self):
        """Reset to primary API key."""
        if self.current_api_key == self.api_key_secondary:
            self.current_api_key = self.api_key
            self.logger.info("Reset to primary API key")
    
    def get_api_key_status(self) -> Dict[str, Any]:
        """Get current API key status for debugging."""
        return {
            "current_key": "primary" if self.current_api_key == self.api_key else "secondary",
            "primary_available": bool(self.api_key),
            "secondary_available": bool(self.api_key_secondary),
            "current_key_preview": f"{self.current_api_key[:8]}..." if self.current_api_key else "None"
        }
    
    def _has_secondary_key(self) -> bool:
        """Check if secondary API key is available."""
        return bool(self.api_key_secondary)
    
    def _is_using_secondary_key(self) -> bool:
        """Check if currently using secondary API key."""
        return self.current_api_key == self.api_key_secondary
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make a robust request to the Gemini API with comprehensive error handling."""
        url = f"{self.base_url}/{endpoint}"
        
        # Log request details for debugging
        self.logger.debug(f"Making request to: {endpoint}")
        self.logger.debug(f"Request data keys: {list(data.keys())}")
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=self._get_headers(),
                    json=data,
                    timeout=60
                )
                
                # Log response details
                self.logger.debug(f"Response status: {response.status_code}")
                self.logger.debug(f"Response headers: {dict(response.headers)}")
                
                if response.status_code == 403:
                    if attempt == 0 and self.current_api_key == self.api_key and self._has_secondary_key():
                        self.logger.warning("Primary API key failed, switching to secondary")
                        self._switch_api_key()
                        continue
                    else:
                        if not self._has_secondary_key():
                            self.logger.error("Primary API key failed and no secondary key available")
                        else:
                            self.logger.error("API key authentication failed for both keys")
                        return None
                
                # Handle rate limiting
                if response.status_code == 429:
                    if attempt == 0 and self.current_api_key == self.api_key and self._has_secondary_key():
                        self.logger.warning("Primary API key rate limited, switching to secondary")
                        self._switch_api_key()
                        continue
                    else:
                        # Both keys are rate limited or no secondary key, wait and retry
                        retry_after = int(response.headers.get('Retry-After', 60))
                        if self._has_secondary_key():
                            self.logger.warning(f"Both API keys rate limited. Waiting {retry_after} seconds before retry...")
                        else:
                            self.logger.warning(f"Primary API key rate limited and no secondary key. Waiting {retry_after} seconds before retry...")
                        time.sleep(retry_after)
                        continue
                
                # Handle other client errors
                if 400 <= response.status_code < 500:
                    error_detail = response.text
                    # Try switching API key for certain client errors
                    if attempt == 0 and self.current_api_key == self.api_key and self._has_secondary_key() and response.status_code in [400, 401, 404]:
                        self.logger.warning(f"Primary API key failed with {response.status_code}, switching to secondary")
                        self._switch_api_key()
                        continue
                    else:
                        self.logger.error(f"Client error {response.status_code}: {error_detail}")
                        return None
                
                # Handle server errors
                if response.status_code >= 500:
                    if attempt == 0 and self.current_api_key == self.api_key and self._has_secondary_key():
                        self.logger.warning(f"Primary API key server error {response.status_code}, switching to secondary")
                        self._switch_api_key()
                        continue
                    else:
                        # Both keys experiencing server errors or no secondary key, wait and retry
                        wait_time = self.rate_limit_delay * (2 ** attempt)
                        if self._has_secondary_key():
                            self.logger.warning(f"Both API keys server error {response.status_code}. Waiting {wait_time}s before retry...")
                        else:
                            self.logger.warning(f"Primary API key server error {response.status_code} and no secondary key. Waiting {wait_time}s before retry...")
                        if attempt < self.max_retries - 1:
                            time.sleep(wait_time)
                            continue
                        else:
                            if self._has_secondary_key():
                                self.logger.error("Server error persisted after all retries with both API keys")
                            else:
                                self.logger.error("Server error persisted after all retries with primary API key")
                            return None
                
                response.raise_for_status()
                
                # Parse JSON response with error handling
                try:
                    json_response = response.json()
                    self.logger.debug(f"Response parsed successfully. Keys: {list(json_response.keys())}")
                    return json_response
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON response: {e}")
                    self.logger.error(f"Response text: {response.text[:500]}...")
                    return None
                
            except requests.exceptions.Timeout as e:
                self.logger.warning(f"Request timeout (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay * (2 ** attempt))
                else:
                    return None
            except requests.exceptions.ConnectionError as e:
                self.logger.warning(f"Connection error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay * (2 ** attempt))
                else:
                    return None
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay * (2 ** attempt))
                else:
                    return None
            except Exception as e:
                self.logger.error(f"Unexpected error during request (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.rate_limit_delay * (2 ** attempt))
                else:
                    return None
        
        return None
    
    def generate_text(self, prompt: str, config: GeminiConfig = None) -> Dict[str, Any]:
        """
        Generate text using Gemini with robust error handling.
        
        Args:
            prompt: Input prompt
            config: Gemini configuration
            
        Returns:
            Generated text response
        """
        try:
            if not config:
                config = GeminiConfig()
            
            if not prompt or not prompt.strip():
                return {"success": False, "error": "Empty or invalid prompt provided"}
            
            self.logger.debug(f"Generating text with model: {config.model}")
            self.logger.debug(f"Prompt length: {len(prompt)} characters")
            
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": config.temperature,
                    "maxOutputTokens": config.max_tokens,
                    "topP": config.top_p,
                    "topK": config.top_k
                }
            }
            
            if config.safety_settings:
                data["safetySettings"] = config.safety_settings
            
            response = self._make_request(f"models/{config.model}:generateContent", data)
            
            if not response:
                return {"success": False, "error": "No response received from API"}
            
            # Robust response parsing
            if 'candidates' not in response:
                self.logger.error(f"Invalid response format: {response}")
                return {"success": False, "error": "Invalid response format - no candidates"}
            
            if not response['candidates']:
                self.logger.error("No candidates in response")
                return {"success": False, "error": "No candidates in response"}
            
            candidate = response['candidates'][0]
            
            # Check for finish reason
            finish_reason = candidate.get('finishReason', '')
            if finish_reason == 'SAFETY':
                return {"success": False, "error": "Content blocked by safety filters"}
            elif finish_reason == 'RECITATION':
                return {"success": False, "error": "Content blocked due to recitation policy"}
            elif finish_reason == 'OTHER':
                return {"success": False, "error": "Content generation failed for unknown reason"}
            
            if 'content' not in candidate:
                self.logger.error(f"Candidate missing content: {candidate}")
                return {"success": False, "error": "No content in candidate response"}
            
            if 'parts' not in candidate['content']:
                self.logger.error(f"Content missing parts: {candidate['content']}")
                return {"success": False, "error": "No parts in content"}
            
            if not candidate['content']['parts']:
                self.logger.error("Empty parts in content")
                return {"success": False, "error": "Empty content parts"}
            
            text_part = candidate['content']['parts'][0]
            if 'text' not in text_part:
                self.logger.error(f"Part missing text: {text_part}")
                return {"success": False, "error": "No text in content part"}
            
            generated_text = text_part['text']
            if not generated_text or not generated_text.strip():
                return {"success": False, "error": "Generated text is empty"}
            
            result = {
                "success": True,
                "text": generated_text,
                "usage": response.get('usageMetadata', {}),
                "safety_ratings": candidate.get('safetyRatings', []),
                "finish_reason": finish_reason
            }
            
            self.logger.debug(f"Text generation successful. Length: {len(generated_text)} characters")
            return result
            
        except Exception as e:
            self.logger.error(f"Unexpected error in generate_text: {e}")
            return {"success": False, "error": f"Text generation failed: {str(e)}"}
    
    def generate_with_functions(self, prompt: str, functions: List[Dict[str, Any]], 
                              config: GeminiConfig = None, enable_google_search: bool = False) -> Dict[str, Any]:
        """
        Generate text with function calling capabilities and optional Google Search.
        
        Args:
            prompt: Input prompt
            functions: List of function definitions
            config: Gemini configuration
            enable_google_search: Whether to enable Google Search tool
            
        Returns:
            Response with potential function calls
        """
        if not config:
            config = GeminiConfig()
        
        tools = [{"functionDeclarations": functions}]
        
        # Add Google Search tool if enabled
        if enable_google_search:
            tools.append({"google_search": {}})
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": tools,
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": config.max_tokens,
                "topP": config.top_p,
                "topK": config.top_k
            }
        }
        
        response = self._make_request(f"models/{config.model}:generateContent", data)
        
        if response and 'candidates' in response:
            candidate = response['candidates'][0]
            if 'content' in candidate:
                content = candidate['content']['parts'][0]
                
                result = {
                    "success": True,
                    "text": content.get('text', ''),
                    "usage": response.get('usageMetadata', {})
                }
                
                # Add function call if present
                if 'functionCall' in content:
                    result["function_call"] = content['functionCall']
                
                # Add grounding metadata if Google Search was used
                if enable_google_search and 'groundingMetadata' in candidate:
                    result["grounding_metadata"] = candidate['groundingMetadata']
                
                return result
        
        return {"success": False, "error": "Failed to generate with functions"}
    
    def generate_with_context_caching(self, prompt: str, context_key: str, 
                                    config: GeminiConfig = None) -> Dict[str, Any]:
        """
        Generate text with context caching.
        
        Args:
            prompt: Input prompt
            context_key: Key for context caching
            config: Gemini configuration
            
        Returns:
            Generated text response
        """
        if not self.cache_enabled:
            return self.generate_text(prompt, config)
        
        # Check cache first
        cache_key = f"{context_key}_{hash(prompt)}"
        if cache_key in self.context_cache:
            cached_response = self.context_cache[cache_key]
            if time.time() - cached_response['timestamp'] < 3600:  # 1 hour cache
                self.logger.debug("Using cached response")
                return cached_response['response']
        
        # Generate new response
        response = self.generate_text(prompt, config)
        
        # Cache the response
        if response.get("success"):
            self.context_cache[cache_key] = {
                'response': response,
                'timestamp': time.time()
            }
        
        return response
    
    def create_session(self, session_id: str = None) -> str:
        """
        Create a new session for conversation continuity.
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        if not session_id:
            session_id = f"session_{int(time.time())}"
        
        self.current_session = {
            'id': session_id,
            'created_at': time.time(),
            'messages': [],
            'context': {}
        }
        
        self.session_history.append(self.current_session)
        self.logger.info(f"Created session: {session_id}")
        return session_id
    
    def add_to_session(self, role: str, content: str, session_id: str = None):
        """
        Add a message to the current session.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            session_id: Optional session ID
        """
        if not session_id:
            session_id = self.current_session['id'] if self.current_session else None
        
        if not session_id:
            self.logger.error("No active session")
            return
        
        message = {
            'role': role,
            'content': content,
            'timestamp': time.time()
        }
        
        for session in self.session_history:
            if session['id'] == session_id:
                session['messages'].append(message)
                break
    
    def generate_with_session(self, prompt: str, session_id: str = None, 
                            config: GeminiConfig = None) -> Dict[str, Any]:
        """
        Generate text using session context.
        
        Args:
            prompt: Input prompt
            session_id: Session ID
            config: Gemini configuration
            
        Returns:
            Generated text response
        """
        if not session_id:
            session_id = self.current_session['id'] if self.current_session else None
        
        if not session_id:
            return self.generate_text(prompt, config)
        
        # Find session
        session = None
        for s in self.session_history:
            if s['id'] == session_id:
                session = s
                break
        
        if not session:
            return {"success": False, "error": "Session not found"}
        
        # Build conversation context
        messages = []
        for msg in session['messages'][-10:]:  # Last 10 messages
            messages.append({
                "role": msg['role'],
                "parts": [{"text": msg['content']}]
            })
        
        # Add current prompt
        messages.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        if not config:
            config = GeminiConfig()
        
        data = {
            "contents": messages,
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": config.max_tokens,
                "topP": config.top_p,
                "topK": config.top_k
            }
        }
        
        response = self._make_request(f"models/{config.model}:generateContent", data)
        
        if response and 'candidates' in response:
            candidate = response['candidates'][0]
            if 'content' in candidate:
                generated_text = candidate['content']['parts'][0]['text']
                
                # Add to session
                self.add_to_session("user", prompt, session_id)
                self.add_to_session("assistant", generated_text, session_id)
                
                return {
                    "success": True,
                    "text": generated_text,
                    "usage": response.get('usageMetadata', {}),
                    "session_id": session_id
                }
        
        return {"success": False, "error": "Failed to generate with session"}
    
    def analyze_image(self, image_path: str, prompt: str = "Describe this image", 
                     config: GeminiConfig = None) -> Dict[str, Any]:
        """
        Analyze an image using Gemini Vision with robust error handling.
        
        Args:
            image_path: Path to image file
            prompt: Analysis prompt
            config: Gemini configuration
            
        Returns:
            Image analysis response
        """
        try:
            import base64
            import os
            
            # Validate inputs
            if not image_path or not image_path.strip():
                return {"success": False, "error": "No image path provided"}
            
            if not prompt or not prompt.strip():
                prompt = "Describe this image"
            
            if not config:
                config = GeminiConfig()
            
            # Check if file exists
            if not os.path.exists(image_path):
                return {"success": False, "error": f"Image file not found: {image_path}"}
            
            # Check if it's a file (not directory)
            if not os.path.isfile(image_path):
                return {"success": False, "error": f"Path is not a file: {image_path}"}
            
            # Check file size (limit to 20MB for Gemini)
            file_size = os.path.getsize(image_path)
            max_size = 20 * 1024 * 1024  # 20MB
            if file_size > max_size:
                return {"success": False, "error": f"Image file too large: {file_size / (1024*1024):.1f}MB (max 20MB)"}
            
            self.logger.debug(f"Analyzing image: {image_path}")
            self.logger.debug(f"File size: {file_size / 1024:.1f}KB")
            self.logger.debug(f"Prompt: {prompt}")
            
            # Read and encode image
            try:
                with open(image_path, 'rb') as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
            except IOError as e:
                return {"success": False, "error": f"Cannot read image file: {e}"}
            except Exception as e:
                return {"success": False, "error": f"Error reading image: {e}"}
            
            # Determine image type with more comprehensive support
            image_type = "image/jpeg"  # default
            file_ext = os.path.splitext(image_path.lower())[1]
            type_mapping = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp',
                '.tiff': 'image/tiff',
                '.tif': 'image/tiff'
            }
            image_type = type_mapping.get(file_ext, 'image/jpeg')
            
            self.logger.debug(f"Image type: {image_type}")
            self.logger.debug(f"Encoded data length: {len(image_data)} characters")
            
            data = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": image_type,
                                "data": image_data
                            }
                        }
                    ]
                }],
                "generationConfig": {
                    "temperature": config.temperature,
                    "maxOutputTokens": config.max_tokens,
                    "topP": config.top_p,
                    "topK": config.top_k
                }
            }
            
            if config.safety_settings:
                data["safetySettings"] = config.safety_settings
            
            response = self._make_request("models/gemini-2.0-flash:generateContent", data)
            
            if not response:
                return {"success": False, "error": "No response received from API"}
            
            # Robust response parsing
            if 'candidates' not in response:
                self.logger.error(f"Invalid response format: {response}")
                return {"success": False, "error": "Invalid response format - no candidates"}
            
            if not response['candidates']:
                self.logger.error("No candidates in response")
                return {"success": False, "error": "No candidates in response"}
            
            candidate = response['candidates'][0]
            
            # Check for finish reason
            finish_reason = candidate.get('finishReason', '')
            if finish_reason == 'SAFETY':
                return {"success": False, "error": "Image analysis blocked by safety filters"}
            elif finish_reason == 'RECITATION':
                return {"success": False, "error": "Image analysis blocked due to recitation policy"}
            elif finish_reason == 'OTHER':
                return {"success": False, "error": "Image analysis failed for unknown reason"}
            
            if 'content' not in candidate:
                self.logger.error(f"Candidate missing content: {candidate}")
                return {"success": False, "error": "No content in candidate response"}
            
            if 'parts' not in candidate['content']:
                self.logger.error(f"Content missing parts: {candidate['content']}")
                return {"success": False, "error": "No parts in content"}
            
            if not candidate['content']['parts']:
                self.logger.error("Empty parts in content")
                return {"success": False, "error": "Empty content parts"}
            
            text_part = candidate['content']['parts'][0]
            if 'text' not in text_part:
                self.logger.error(f"Part missing text: {text_part}")
                return {"success": False, "error": "No text in content part"}
            
            analysis_text = text_part['text']
            if not analysis_text or not analysis_text.strip():
                return {"success": False, "error": "Image analysis result is empty"}
            
            result = {
                "success": True,
                "text": analysis_text,
                "usage": response.get('usageMetadata', {}),
                "safety_ratings": candidate.get('safetyRatings', []),
                "finish_reason": finish_reason,
                "image_info": {
                    "path": image_path,
                    "size": file_size,
                    "type": image_type
                }
            }
            
            self.logger.debug(f"Image analysis successful. Result length: {len(analysis_text)} characters")
            return result
            
        except Exception as e:
            self.logger.error(f"Unexpected error in analyze_image: {e}")
            return {"success": False, "error": f"Image analysis failed: {str(e)}"}
    
    def generate_structured_output(self, prompt: str, schema: Dict[str, Any], 
                                 config: GeminiConfig = None) -> Dict[str, Any]:
        """
        Generate structured output using JSON schema.
        
        Args:
            prompt: Input prompt
            schema: JSON schema for output
            config: Gemini configuration
            
        Returns:
            Structured response
        """
        structured_prompt = f"""
{prompt}

Please respond with a JSON object that matches this schema:
{json.dumps(schema, indent=2)}

Return only the JSON object, no additional text.
"""
        
        response = self.generate_text(structured_prompt, config)
        
        if response.get("success"):
            try:
                # Try to parse JSON from response
                text = response["text"]
                # Extract JSON from response if it's wrapped in markdown
                if "```json" in text:
                    json_start = text.find("```json") + 7
                    json_end = text.find("```", json_start)
                    json_text = text[json_start:json_end].strip()
                else:
                    json_text = text.strip()
                
                structured_data = json.loads(json_text)
                return {
                    "success": True,
                    "data": structured_data,
                    "raw_text": text
                }
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "error": f"Failed to parse structured output: {e}",
                    "raw_text": response["text"]
                }
        
        return response
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available Gemini models."""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._get_headers(),
                timeout=10
            )
            response.raise_for_status()
            return {"success": True, "models": response.json()}
        except Exception as e:
            return {"success": False, "error": f"Failed to get models: {e}"}
    
    def clear_cache(self):
        """Clear the context cache."""
        self.context_cache.clear()
        self.logger.info("Context cache cleared")
    
    def generate_with_google_search(self, prompt: str, config: GeminiConfig = None) -> Dict[str, Any]:
        """
        Generate text with Google Search grounding using the official tool.
        
        Args:
            prompt: Input prompt
            config: Gemini configuration
            
        Returns:
            Response with grounded information and citations
        """
        if not config:
            config = GeminiConfig()
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {
                "temperature": config.temperature,
                "maxOutputTokens": config.max_tokens,
                "topP": config.top_p,
                "topK": config.top_k
            }
        }
        
        response = self._make_request(f"models/{config.model}:generateContent", data)
        
        if response and 'candidates' in response:
            candidate = response['candidates'][0]
            if 'content' in candidate:
                content = candidate['content']['parts'][0]
                grounding_metadata = candidate.get('groundingMetadata', {})
                
                return {
                    "success": True,
                    "text": content['text'],
                    "usage": response.get('usageMetadata', {}),
                    "grounding_metadata": grounding_metadata,
                    "web_search_queries": grounding_metadata.get('webSearchQueries', []),
                    "grounding_chunks": grounding_metadata.get('groundingChunks', []),
                    "grounding_supports": grounding_metadata.get('groundingSupports', [])
                }
        
        return {"success": False, "error": "Failed to generate with Google Search"}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_enabled": self.cache_enabled,
            "cached_items": len(self.context_cache),
            "cache_size_mb": sum(len(str(item)) for item in self.context_cache.values()) / 1024 / 1024
        }
    
    def enhanced_web_search(self, query: str, adaptive: bool = True) -> Dict[str, Any]:
        """Perform web search with adaptive result length based on query complexity."""
        try:
            # Validate input
            if not query or not query.strip():
                return {"success": False, "error": "Empty or invalid search query provided"}
            
            # Determine query complexity and appropriate response parameters
            complexity = self._determine_query_complexity(query)
            
            self.logger.info(f"🔍 Searching for: {query}")
            self.logger.info(f"📊 Query complexity: {complexity['description']} (sources: {complexity['sources']}, tokens: {complexity['max_tokens']})")
            start_time = time.time()
            
            # Use Google Search grounding for real-time information
            try:
                import google.generativeai as genai
            except ImportError as e:
                self.logger.error(f"Google Search dependencies not available: {e}")
                return {"success": False, "error": "Google Search not available - dependencies not installed"}
            
            # Configure Gemini with Google Search
            try:
                genai.configure(api_key=self.current_api_key)
            except Exception as e:
                self.logger.error(f"Failed to configure Gemini: {e}")
                return {"success": False, "error": f"Failed to configure search client: {str(e)}"}
            
            # Create a dynamic prompt based on query complexity
            if complexity['level'] == 'simple':
                response_style = "Provide a concise, direct answer with key facts and essential information."
                source_instruction = f"Use approximately {complexity['sources']} high-quality sources for accuracy."
            elif complexity['level'] == 'complex':
                response_style = "Provide a detailed analysis with clear explanations, examples, and comprehensive coverage."
                source_instruction = f"Use approximately {complexity['sources']} comprehensive sources for thorough analysis."
            elif complexity['level'] == 'technical':
                response_style = "Provide an in-depth, technical analysis with detailed explanations, methodologies, and comprehensive research."
                source_instruction = f"Use approximately {complexity['sources']} authoritative sources for complete technical coverage."
            else:  # moderate
                response_style = "Provide a balanced, well-structured response with good detail and clear organization."
                source_instruction = f"Use approximately {complexity['sources']} reliable sources for comprehensive coverage."

            enhanced_query = f"""Please answer: {query}

Response Requirements:
{response_style}

Source Requirements:
{source_instruction}

Please provide a comprehensive, well-researched answer using the available tools and sources."""

            # Perform search with grounding using the existing method
            try:
                response = self.generate_with_google_search(enhanced_query)
                if not response.get("success"):
                    return response
            except Exception as e:
                self.logger.error(f"Search request failed: {e}")
                return {"success": False, "error": f"Search request failed: {str(e)}"}
            
            search_time = time.time() - start_time
            
            # Robust response validation
            if not response:
                self.logger.error("No response received from search")
                return {"success": False, "error": "No response received from search"}
            
            # Extract response with validation
            result = response.get("text", "").strip()
            if not result:
                self.logger.error("Search response is empty after stripping")
                return {"success": False, "error": "Search response is empty"}
            
            # Get usage metadata with validation
            usage_metadata = response.get("usage", {})
            
            self.logger.success(f"✅ Search completed in {search_time:.2f} seconds")
            
            # Add complexity and source information
            complexity_info = f"\n\n📊 Search Analysis:\n- Query Complexity: {complexity['description']}\n- Target Sources: {complexity['sources']}\n- Max Tokens: {complexity['max_tokens']}\n- Search Time: {search_time:.2f}s"
            
            # Combine result with complexity info
            final_result = result + complexity_info
            
            return {
                "success": True,
                "output": final_result,
                "query": query,
                "complexity": complexity,
                "usage_metadata": usage_metadata,
                "search_time": search_time,
                "response_length": len(result)
            }
            
        except Exception as e:
            self.logger.error(f"Unexpected error in enhanced_web_search: {e}")
            return {"success": False, "error": f"Enhanced search failed: {str(e)}"}

    def _determine_query_complexity(self, query: str) -> dict:
        """Analyze query to determine appropriate response length and source count."""
        query_lower = query.lower()
        
        # Simple factual queries (1-2 sources, shorter response)
        simple_indicators = [
            "what is", "who is", "when is", "where is", "how much", "how many",
            "define", "meaning of", "definition", "simple", "quick", "brief"
        ]
        
        # Complex analytical queries (5-10 sources, longer response)
        complex_indicators = [
            "compare", "analyze", "analysis", "research", "comprehensive",
            "detailed", "in-depth", "explain", "discuss", "evaluate",
            "pros and cons", "advantages and disadvantages", "differences between"
        ]
        
        # Research-intensive queries (8-15 sources, very detailed response)
        research_indicators = [
            "comprehensive analysis", "extensive research", "multiple perspectives",
            "various approaches", "different methods", "complete overview",
            "thorough investigation", "detailed comparison", "in-depth study"
        ]
        
        # Technical/specialized queries (10-20 sources, highly detailed)
        technical_indicators = [
            "technical", "implementation", "architecture", "methodology",
            "framework", "algorithm", "protocol", "specification", "standard"
        ]
        
        # Count indicators
        simple_count = sum(1 for indicator in simple_indicators if indicator in query_lower)
        complex_count = sum(1 for indicator in complex_indicators if indicator in query_lower)
        research_count = sum(1 for indicator in research_indicators if indicator in query_lower)
        technical_count = sum(1 for indicator in technical_indicators if indicator in query_lower)
        
        # Determine complexity level
        if technical_count > 0 or research_count > 0:
            return {
                "level": "technical",
                "sources": 15,
                "max_tokens": 16384,
                "description": "Technical/Research query - comprehensive analysis"
            }
        elif complex_count > 0:
            return {
                "level": "complex",
                "sources": 8,
                "max_tokens": 12288,
                "description": "Complex query - detailed analysis"
            }
        elif simple_count > 0:
            return {
                "level": "simple",
                "sources": 3,
                "max_tokens": 4096,
                "description": "Simple query - concise answer"
            }
        else:
            # Default to moderate complexity
            return {
                "level": "moderate",
                "sources": 5,
                "max_tokens": 8192,
                "description": "Moderate query - balanced response"
            }

    def analyze_urls(self, urls: list) -> Dict[str, Any]:
        """Analyze multiple URLs and extract information."""
        try:
            self.logger.info(f"🔗 Analyzing {len(urls)} URLs")
            start_time = time.time()
            
            # Create analysis prompt
            analysis_prompt = f"""Please analyze the following URLs and provide comprehensive information about each:

URLs to analyze: {', '.join(urls)}

For each URL, provide:
1. Main topic and content summary
2. Key information and insights
3. Relevance and credibility
4. Main points and takeaways
5. Any important details or context

Please structure your response clearly for each URL."""

            # Perform URL analysis using existing method
            response = self.generate_text(analysis_prompt)
            
            analysis_time = time.time() - start_time
            self.logger.success(f"✅ URL analysis completed in {analysis_time:.2f} seconds")
            
            # Extract response
            result = response.get("text", "") if response.get("success") else ""
            
            return {
                "success": True,
                "output": result,
                "urls": urls,
                "analysis_time": analysis_time,
                "url_count": len(urls)
            }
            
        except Exception as e:
            self.logger.error(f"URL analysis failed: {e}")
            return {"success": False, "error": f"URL analysis failed: {e}"}
