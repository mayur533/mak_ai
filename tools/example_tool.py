"""
Example tool for the AI Assistant System.
Demonstrates how to create custom tools that the AI can use.
"""

import json
from datetime import datetime
from pathlib import Path


def example_tool(message: str, save_to_file: bool = False):
    """
    An example tool that processes a message and optionally saves it to a file.
    
    Args:
        message: The message to process
        save_to_file: Whether to save the result to a file
        
    Returns:
        Dict with success status and result
    """
    try:
        # Process the message
        result = {
            "original_message": message,
            "processed_at": datetime.now().isoformat(),
            "word_count": len(message.split()),
            "character_count": len(message),
            "uppercase": message.upper(),
            "lowercase": message.lower()
        }
        
        # Save to file if requested
        if save_to_file:
            output_file = Path("output") / f"example_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_file.parent.mkdir(exist_ok=True)
            
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            
            result["saved_to"] = str(output_file)
        
        return {
            "success": True,
            "output": f"Processed message: '{message}'. Word count: {result['word_count']}, Character count: {result['character_count']}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Example tool failed: {str(e)}"
        }
