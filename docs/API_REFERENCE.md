# AI Assistant System - API Reference

## Overview

The AI Assistant System is a modular, self-improving AI agent with comprehensive tool management, context awareness, and dynamic capabilities.

## Core Components

### AISystem

The main orchestrator class that handles AI interactions, tool management, and task execution.

#### Methods

##### `__init__(voice_mode: bool = None)`
Initialize the AI system.

**Parameters:**
- `voice_mode` (bool, optional): Enable voice mode for speech input/output

**Example:**
```python
from src.core.ai_system import AISystem

ai_system = AISystem(voice_mode=False)
```

##### `process_request(user_input: str)`
Process a user request and execute the appropriate plan.

**Parameters:**
- `user_input` (str): The user's request or command

**Returns:**
- `bool`: True if processing completed successfully

**Example:**
```python
result = ai_system.process_request("play music on youtube")
```

##### `run()`
Start the interactive AI system loop.

**Example:**
```python
ai_system.run()
```

##### `shutdown()`
Gracefully shutdown the AI system.

**Example:**
```python
ai_system.shutdown()
```

## Tools API

### BaseTools

Core functionality tools that are always available.

#### File Operations

##### `read_file(file_path: str) -> Dict[str, Any]`
Read the contents of a file.

**Parameters:**
- `file_path` (str): Path to the file to read

**Returns:**
- `Dict[str, Any]`: Result with success status and file content

**Example:**
```python
result = base_tools.read_file("example.txt")
if result["success"]:
    print(result["output"])
```

##### `write_file(file_path: str, content: str) -> Dict[str, Any]`
Write content to a file.

**Parameters:**
- `file_path` (str): Path where to write the file
- `content` (str): Content to write

**Returns:**
- `Dict[str, Any]`: Result with success status

**Example:**
```python
result = base_tools.write_file("output.txt", "Hello World")
```

##### `list_dir(directory: str = ".") -> Dict[str, Any]`
List contents of a directory.

**Parameters:**
- `directory` (str): Directory path to list

**Returns:**
- `Dict[str, Any]`: Result with directory contents

**Example:**
```python
result = base_tools.list_dir("/home/user")
```

##### `change_dir(directory: str) -> Dict[str, Any]`
Change the current working directory.

**Parameters:**
- `directory` (str): Directory path to change to

**Returns:**
- `Dict[str, Any]`: Result with success status

**Example:**
```python
result = base_tools.change_dir("/home/user")
```

#### System Operations

##### `run_shell(command: str) -> Dict[str, Any]`
Execute a shell command with security validation.

**Parameters:**
- `command` (str): Shell command to execute

**Returns:**
- `Dict[str, Any]`: Result with command output

**Example:**
```python
result = base_tools.run_shell("ls -la")
```

##### `install_package(package_name: str) -> Dict[str, Any]`
Install a Python package in the virtual environment.

**Parameters:**
- `package_name` (str): Name of the package to install

**Returns:**
- `Dict[str, Any]`: Result with installation status

**Example:**
```python
result = base_tools.install_package("requests")
```

##### `check_system_dependency(dependency_name: str) -> Dict[str, Any]`
Check if a system dependency is installed.

**Parameters:**
- `dependency_name` (str): Name of the dependency to check

**Returns:**
- `Dict[str, Any]`: Result with dependency status and installation instructions

**Example:**
```python
result = base_tools.check_system_dependency("xdotool")
```

#### GUI Automation

##### `read_screen(prompt: str = "Describe what you see on this screen") -> Dict[str, Any]`
Capture and analyze the current screen.

**Parameters:**
- `prompt` (str): Analysis prompt for the screen capture

**Returns:**
- `Dict[str, Any]`: Result with screen analysis

**Example:**
```python
result = base_tools.read_screen("What applications are open?")
```

##### `click_screen(x: int, y: int, button: str = "left") -> Dict[str, Any]`
Click at specific screen coordinates.

**Parameters:**
- `x` (int): X coordinate
- `y` (int): Y coordinate
- `button` (str): Mouse button ("left", "right", "middle")

**Returns:**
- `Dict[str, Any]`: Result with click status

**Example:**
```python
result = base_tools.click_screen(100, 200)
```

##### `analyze_screen_actions(task_description: str) -> Dict[str, Any]`
Analyze screen and provide actionable steps with coordinates.

**Parameters:**
- `task_description` (str): Description of the task to perform

**Returns:**
- `Dict[str, Any]`: Result with actionable steps and coordinates

**Example:**
```python
result = base_tools.analyze_screen_actions("Click on the search button")
```

#### Data Processing

##### `read_json_file(file_path: str) -> Dict[str, Any]`
Read and parse a JSON file.

**Parameters:**
- `file_path` (str): Path to the JSON file

**Returns:**
- `Dict[str, Any]`: Result with parsed JSON data

**Example:**
```python
result = base_tools.read_json_file("config.json")
```

##### `write_json_file(file_path: str, data: Any, indent: int = 2) -> Dict[str, Any]`
Write data to a JSON file.

**Parameters:**
- `file_path` (str): Path where to write the JSON file
- `data` (Any): Data to serialize to JSON
- `indent` (int): JSON indentation level

**Returns:**
- `Dict[str, Any]`: Result with write status

**Example:**
```python
data = {"name": "John", "age": 30}
result = base_tools.write_json_file("user.json", data)
```

### GeminiClient

Enhanced Gemini API client with advanced features.

#### Methods

##### `generate_text(prompt: str, **kwargs) -> str`
Generate text using the Gemini API.

**Parameters:**
- `prompt` (str): Text prompt for generation
- `**kwargs`: Additional generation parameters

**Returns:**
- `str`: Generated text

**Example:**
```python
response = gemini_client.generate_text("Write a poem about AI")
```

##### `analyze_image(image_path: str, prompt: str = "Describe this image") -> Dict[str, Any]`
Analyze an image using Gemini Vision.

**Parameters:**
- `image_path` (str): Path to the image file
- `prompt` (str): Analysis prompt

**Returns:**
- `Dict[str, Any]`: Result with image analysis

**Example:**
```python
result = gemini_client.analyze_image("photo.jpg", "What objects are in this image?")
```

##### `enhanced_web_search(query: str, adaptive: bool = True) -> str`
Perform web search with adaptive result length.

**Parameters:**
- `query` (str): Search query
- `adaptive` (bool): Use adaptive result length

**Returns:**
- `str`: Search results

**Example:**
```python
results = gemini_client.enhanced_web_search("latest AI developments")
```

## Configuration

### Settings

Configuration management for the AI Assistant System.

#### Environment Variables

- `GEMINI_API_KEY`: Primary Gemini API key
- `GEMINI_API_KEY_SECONDARY`: Secondary Gemini API key for fallback
- `GEMINI_MODEL`: Gemini model to use (default: gemini-2.5-flash)
- `VOICE_ENABLED`: Enable voice mode (default: false)
- `DEBUG_MODE`: Enable debug mode (default: false)
- `LOG_LEVEL`: Logging level (default: INFO)
- `MAX_RETRIES`: Maximum retry attempts (default: 3)
- `DB_PATH`: Database file path (default: db/agent_memory.db)

#### Example .env file

```env
GEMINI_API_KEY=your_primary_api_key_here
GEMINI_API_KEY_SECONDARY=your_secondary_api_key_here
GEMINI_MODEL=gemini-2.5-flash
VOICE_ENABLED=false
DEBUG_MODE=false
LOG_LEVEL=INFO
MAX_RETRIES=3
DB_PATH=db/agent_memory.db
```

## Error Handling

All tools return a standardized result dictionary:

```python
{
    "success": bool,           # Whether the operation succeeded
    "output": str,            # Success message or output
    "error": str,             # Error message if failed
    "message": str,           # Additional information
    "data": Any               # Additional data (tool-specific)
}
```

### Common Error Codes

- `"INVALID_INPUT"`: Invalid input parameters
- `"FILE_NOT_FOUND"`: File or directory not found
- `"PERMISSION_DENIED"`: Insufficient permissions
- `"TOOL_NOT_FOUND"`: Tool does not exist
- `"API_ERROR"`: API call failed
- `"TIMEOUT"`: Operation timed out
- `"VALIDATION_ERROR"`: Input validation failed

## Best Practices

### Tool Development

1. **Always return standardized results**: Use the standard result dictionary format
2. **Validate inputs**: Check input parameters before processing
3. **Handle errors gracefully**: Provide meaningful error messages
4. **Use type hints**: Add type annotations for better code clarity
5. **Document parameters**: Provide clear parameter descriptions

### Security

1. **Validate user input**: Always sanitize and validate user inputs
2. **Use safe file paths**: Prevent directory traversal attacks
3. **Limit system access**: Restrict access to system directories
4. **Validate commands**: Check shell commands for dangerous patterns

### Performance

1. **Use async operations**: For I/O intensive tasks
2. **Implement caching**: Cache frequently accessed data
3. **Monitor resources**: Track CPU, memory, and disk usage
4. **Optimize database queries**: Use efficient database operations

## Examples

### Basic Usage

```python
from src.core.ai_system import AISystem

# Initialize the system
ai_system = AISystem()

# Process a request
ai_system.process_request("create a text file with 'Hello World'")

# Run interactive mode
ai_system.run()
```

### Custom Tool Development

```python
from src.database.memory import Tool

def my_custom_tool(param1: str, param2: int) -> Dict[str, Any]:
    """Custom tool that does something useful."""
    try:
        # Tool logic here
        result = f"Processed {param1} with {param2}"
        
        return {
            "success": True,
            "output": result,
            "message": "Custom tool executed successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "output": "Custom tool failed"
        }

# Register the tool
tool = Tool(
    name="my_custom_tool",
    code="",  # Code is optional for built-in tools
    doc="Custom tool that does something useful. Usage: my_custom_tool(param1, param2)",
    is_dynamic=False,
    func=my_custom_tool
)

ai_system.tool_manager.register_tool(tool)
```

### Async Operations

```python
from src.core.async_ai_system import AsyncAISystem
import asyncio

async def main():
    async with AsyncAISystem() as ai_system:
        result = await ai_system.process_request_async("analyze system performance")
        print(result)

asyncio.run(main())
```

### Monitoring

```python
from src.monitoring import get_health_status, get_metrics_summary

# Get system health
health = get_health_status()
print(f"System status: {health['status']}")

# Get metrics summary
metrics = get_metrics_summary()
print(f"Total requests: {metrics['application']['requests_total']}")
```

## Troubleshooting

### Common Issues

1. **API Key Issues**: Ensure API keys are set in .env file
2. **Permission Errors**: Check file and directory permissions
3. **Tool Not Found**: Verify tool is registered correctly
4. **Memory Issues**: Monitor system resources and context size

### Debug Mode

Enable debug mode for detailed logging:

```bash
python main.py --debug
```

### Logs

Check log files in the `log/` directory for detailed error information.

## Support

For issues and questions:
1. Check the logs for error details
2. Enable debug mode for verbose output
3. Verify configuration settings
4. Check system requirements and dependencies
