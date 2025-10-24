"""
Google Search integration for AI Assistant System.
Provides web search capabilities using Google Search API.
"""

import requests
import json
from typing import Dict, Any, List, Optional
from urllib.parse import quote_plus

# Add parent directory to path for imports
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.config.settings import settings
from src.logging.logger import logger


class GoogleSearchTool:
    """Google Search integration tool."""

    def __init__(self):
        """Initialize Google Search tool."""
        self.logger = logger
        # Use Gemini API key for Google Search (works with Gemini API)
        self.search_api_key = settings.GEMINI_API_KEY
        self.base_url = "https://www.googleapis.com/customsearch/v1"

        if not self.search_api_key:
            self.logger.warning(
                "Google Search requires Gemini API key. Set GEMINI_API_KEY in .env"
            )

    def search(
        self, query: str, num_results: int = 10, safe_search: str = "moderate"
    ) -> Dict[str, Any]:
        """
        Perform a Google search using the official Gemini API Google Search tool.

        Args:
            query: Search query
            num_results: Number of results to return (max 10)
            safe_search: Safe search level (off, moderate, strict)

        Returns:
            Dict with search results
        """
        if not self.search_api_key:
            return {
                "success": False,
                "error": "Google Search requires Gemini API key. Please set GEMINI_API_KEY in .env file",
            }

        try:
            self.logger.info(f"Searching Google for: {query}")

            # Use official Gemini API with Google Search grounding
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": query}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 2048,
                    "topP": 0.95,
                    "topK": 40,
                },
                "tools": [{"google_search": {}}],
            }
            params = {"key": self.search_api_key}

            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                headers=headers,
                json=payload,
                params=params,
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()

            if "candidates" in data and data["candidates"]:
                candidate = data["candidates"][0]
                search_result = candidate["content"]["parts"][0]["text"]

                # Extract grounding metadata for proper search results
                grounding_metadata = candidate.get("groundingMetadata", {})
                web_search_queries = grounding_metadata.get("webSearchQueries", [])
                grounding_chunks = grounding_metadata.get("groundingChunks", [])
                grounding_supports = grounding_metadata.get("groundingSupports", [])

                # Format search results with proper citations
                results = []
                if grounding_chunks:
                    for i, chunk in enumerate(grounding_chunks):
                        web_data = chunk.get("web", {})
                        results.append(
                            {
                                "title": web_data.get("title", f"Search Result {i+1}"),
                                "link": web_data.get("uri", ""),
                                "snippet": f"Source {i+1} from {web_data.get('title', 'Unknown')}",
                                "display_link": web_data.get("title", "Unknown"),
                                "formatted_url": web_data.get("uri", ""),
                                "citation_index": i + 1,
                            }
                        )
                else:
                    # Fallback if no grounding metadata
                    results = [
                        {
                            "title": f"Search Results for: {query}",
                            "link": f"Web search for: {query}",
                            "snippet": search_result,
                            "display_link": "gemini-search",
                            "formatted_url": f"search://{query}",
                            "citation_index": 1,
                        }
                    ]

                # Add inline citations to the response text
                text_with_citations = self._add_inline_citations(
                    search_result, grounding_supports, grounding_chunks
                )

                return {
                    "success": True,
                    "results": results,
                    "total_results": str(len(results)),
                    "search_time": "0.1",
                    "query": query,
                    "raw_response": search_result,
                    "text_with_citations": text_with_citations,
                    "web_search_queries": web_search_queries,
                    "output": text_with_citations,  # Add this so the system displays the actual content
                    "grounding_metadata": grounding_metadata,
                }
            else:
                return {
                    "success": False,
                    "error": "No search results returned from Gemini API",
                }

        except requests.exceptions.RequestException as e:
            error_msg = f"Google Search API request failed: {e}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"Google Search error: {e}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def search_news(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """
        Search for news articles using Gemini API.

        Args:
            query: News search query
            num_results: Number of results to return

        Returns:
            Dict with news search results
        """
        news_query = f"{query} news"
        return self.search(news_query, num_results)

    def search_images(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """
        Search for images using Gemini API.

        Args:
            query: Image search query
            num_results: Number of results to return

        Returns:
            Dict with image search results
        """
        if not self.search_api_key:
            return {"success": False, "error": "Google Search requires Gemini API key"}

        try:
            self.logger.info(f"Searching for images: {query}")

            # Use Gemini API for image search
            search_prompt = f"""
Search for images related to: {query}

Please provide:
1. A description of what images would be relevant for this search
2. Suggestions for image search terms
3. Information about where to find such images

Query: {query}
Number of results requested: {num_results}
"""

            # Use Gemini API for image search
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": search_prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 1024,
                    "topP": 0.95,
                    "topK": 40,
                },
            }
            params = {"key": self.search_api_key}

            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                headers=headers,
                json=payload,
                params=params,
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()

            if "candidates" in data and data["candidates"]:
                search_result = data["candidates"][0]["content"]["parts"][0]["text"]

                # Format as image search results
                results = [
                    {
                        "title": f"Image Search Results for: {query}",
                        "link": f"Image search for: {query}",
                        "snippet": search_result,
                        "display_link": "gemini-image-search",
                        "image_url": f"search://images/{query}",
                        "thumbnail_url": f"search://images/{query}",
                        "width": "unknown",
                        "height": "unknown",
                    }
                ]

                return {
                    "success": True,
                    "results": results,
                    "total_results": "1",
                    "query": query,
                    "raw_response": search_result,
                }
            else:
                return {
                    "success": False,
                    "error": "No image search results returned from Gemini API",
                }

        except Exception as e:
            error_msg = f"Google Images search error: {e}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def search_scholar(self, query: str, num_results: int = 10) -> Dict[str, Any]:
        """
        Search for academic papers and scholarly articles.

        Args:
            query: Academic search query
            num_results: Number of results to return

        Returns:
            Dict with scholarly search results
        """
        scholar_query = f"site:scholar.google.com {query}"
        return self.search(scholar_query, num_results)

    def get_search_suggestions(self, query: str) -> Dict[str, Any]:
        """
        Get search suggestions for a query.

        Args:
            query: Partial search query

        Returns:
            Dict with search suggestions
        """
        try:
            # Use a simple search to get suggestions
            params = {
                "key": self.search_api_key,
                "cx": self.search_engine_id,
                "q": query,
                "num": 1,
            }

            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()

            data = response.json()

            # Extract suggestions from search results
            suggestions = []
            if "items" in data:
                for item in data["items"]:
                    title = item.get("title", "")
                    if title and query.lower() in title.lower():
                        suggestions.append(title)

            return {
                "success": True,
                "suggestions": suggestions[:5],  # Limit to 5 suggestions
                "query": query,
            }

        except Exception as e:
            error_msg = f"Failed to get search suggestions: {e}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}

    def format_search_results(self, results: Dict[str, Any]) -> str:
        """
        Format search results for display.

        Args:
            results: Search results from search method

        Returns:
            Formatted string of search results
        """
        if not results.get("success"):
            return f"Search failed: {results.get('error', 'Unknown error')}"

        if not results.get("results"):
            return "No results found."

        formatted = []
        formatted.append(f"Search Results for: {results.get('query', 'Unknown query')}")
        formatted.append(f"Total Results: {results.get('total_results', 'Unknown')}")
        formatted.append("=" * 50)

        for i, result in enumerate(results["results"], 1):
            formatted.append(f"\n{i}. {result.get('title', 'No title')}")
            formatted.append(f"   URL: {result.get('link', 'No URL')}")
            formatted.append(f"   Snippet: {result.get('snippet', 'No snippet')}")
            if result.get("display_link"):
                formatted.append(f"   Site: {result.get('display_link')}")

        return "\n".join(formatted)

    def is_configured(self) -> bool:
        """Check if Google Search is properly configured."""
        return bool(self.search_api_key)

    def _add_inline_citations(
        self, text: str, grounding_supports: List[Dict], grounding_chunks: List[Dict]
    ) -> str:
        """
        Add inline citations to text based on grounding metadata.

        Args:
            text: Original text
            grounding_supports: Grounding supports from API response
            grounding_chunks: Grounding chunks from API response

        Returns:
            Text with inline citations
        """
        if not grounding_supports or not grounding_chunks:
            return text

        # Sort supports by end_index in descending order to avoid shifting issues
        sorted_supports = sorted(
            grounding_supports,
            key=lambda s: s.get("segment", {}).get("endIndex", 0),
            reverse=True,
        )

        result_text = text

        for support in sorted_supports:
            segment = support.get("segment", {})
            end_index = segment.get("endIndex", 0)
            chunk_indices = support.get("groundingChunkIndices", [])

            if end_index and chunk_indices:
                # Create citation string like [1](link1)[2](link2)
                citation_links = []
                for i in chunk_indices:
                    if i < len(grounding_chunks):
                        chunk = grounding_chunks[i]
                        web_data = chunk.get("web", {})
                        uri = web_data.get("uri", "")
                        if uri:
                            citation_links.append(f"[{i + 1}]({uri})")

                if citation_links:
                    citation_string = ", ".join(citation_links)
                    result_text = (
                        result_text[:end_index]
                        + citation_string
                        + result_text[end_index:]
                    )

        return result_text

    def get_configuration_status(self) -> Dict[str, Any]:
        """Get configuration status."""
        return {
            "api_key_configured": bool(self.search_api_key),
            "fully_configured": self.is_configured(),
            "base_url": self.base_url,
        }
