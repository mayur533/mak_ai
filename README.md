# AI Assistant System

A modular, self-improving AI agent system with dynamic tool creation, memory persistence, and voice capabilities.

## Features

- 🤖 **Autonomous AI Agent**: Self-improving AI that can create and use tools dynamically
- 🛠️ **Dynamic Tool Creation**: Create new tools on-the-fly to solve complex problems
- 🧠 **Memory System**: Persistent memory with SQLite database for long-term learning
- 🎤 **Voice Interface**: Optional speech-to-text and text-to-speech capabilities
- 📁 **Modular Architecture**: Clean, organized codebase with separate modules
- ⚙️ **Configuration Management**: Easy setup with .env files
- 🔧 **Extensible**: Easy to add new tools and functionality

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd ai_assistant_system

# Install dependencies
pip install -r requirements.txt

# Install voice dependencies (optional)
pip install speechrecognition pydub gtts pygame
```

### 2. Configuration

```bash
# Copy the template and add your API keys
cp .env.template .env

# Edit .env file with your Gemini API keys
nano .env
```

Required environment variables:
```env
GEMINI_API_KEY=your_primary_gemini_api_key_here
GEMINI_API_KEY_SECONDARY=your_secondary_gemini_api_key_here
```

### 3. Run the System

```bash
# Text mode (default)
python main.py

# Voice mode
python main.py --voice

# Debug mode
python main.py --debug

# Custom config
python main.py --config custom.env
```

## Project Structure

```
ai_assistant_system/
├── src/                          # Source code
│   ├── config/                   # Configuration management
│   │   └── settings.py          # Settings and environment handling
│   ├── core/                    # Core AI system
│   │   └── ai_system.py         # Main AI orchestrator
│   ├── database/                # Database and memory
│   │   └── memory.py            # Memory management and persistence
│   ├── logging/                 # Logging system
│   │   └── logger.py            # Advanced logging with colors
│   └── tools/                   # Tools and utilities
│       ├── base_tools.py        # Core functionality tools
│       └── voice_tools.py       # Voice input/output tools
├── tools/                       # Dynamic tools directory
├── db/                         # Database files
├── log/                        # Log files
├── output/                     # Output files
├── temp/                       # Temporary files
├── main.py                     # Main entry point
├── requirements.txt            # Python dependencies
├── .env.template              # Environment template
└── README.md                  # This file
```

## Usage Examples

### Basic Text Interaction

```bash
python main.py
> Create a Python script that calculates fibonacci numbers
> Write a tool that can analyze CSV files
> Help me debug this error in my code
```

### Voice Interaction

```bash
python main.py --voice
# Speak your requests and get voice responses
```

### Custom Tools

The system can create and use custom tools dynamically. Tools are automatically saved to the `tools/` directory and loaded on startup.

## Configuration Options

All configuration is handled through the `.env` file:

```env
# API Configuration
GEMINI_API_KEY=your_primary_key
GEMINI_API_KEY_SECONDARY=your_secondary_key
GEMINI_MODEL=gemini-2.5-flash

# System Settings
MAX_HISTORY=100
MAX_RETRIES=3
LOG_LEVEL=INFO

# Voice Settings (optional)
VOICE_ENABLED=false
TTS_LANGUAGE=en
TTS_SPEED=1.0

# Development
DEBUG_MODE=false
VERBOSE_LOGGING=false
```

## Available Tools

### Core Tools
- `run_shell(command)` - Execute shell commands
- `run_python(code)` - Execute Python code
- `read_file(path)` - Read file contents
- `write_file(path, content)` - Write to files
- `list_dir(directory)` - List directory contents
- `install_package(name)` - Install Python packages
- `create_and_save_tool(name, code, doc)` - Create new tools
- `complete_task(message)` - Mark task as complete

### Voice Tools (when enabled)
- `speak(text)` - Convert text to speech
- `listen(timeout)` - Listen for speech input
- `test_voice_system()` - Test voice functionality

## Development

### Adding New Tools

1. Create a new Python file in `src/tools/`
2. Define your tool function
3. The system will automatically load it on startup

Example tool:
```python
def my_custom_tool(param1, param2):
    """My custom tool description."""
    # Tool implementation
    return {"success": True, "output": "Result"}
```

### Extending the System

The modular architecture makes it easy to extend:

- **New Tools**: Add to `src/tools/`
- **Database Features**: Extend `src/database/memory.py`
- **Logging**: Customize `src/logging/logger.py`
- **Configuration**: Add options to `src/config/settings.py`

## Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure your Gemini API keys are valid and set in `.env`
2. **Voice Not Working**: Install voice dependencies: `pip install speechrecognition pydub gtts pygame`
3. **Import Errors**: Make sure you're running from the project root directory
4. **Permission Errors**: Check file/directory permissions for the project folder

### Debug Mode

Run with `--debug` flag for detailed logging:
```bash
python main.py --debug
```

### Logs

Check the `log/` directory for detailed session logs and error information.

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `log/` directory
3. Enable debug mode for detailed information
4. Create an issue with detailed error information
