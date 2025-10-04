"""
Voice tools for the AI Assistant System.
Provides text-to-speech and speech-to-text capabilities.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config.settings import settings
from src.logging.logger import logger

# Voice dependencies (optional)
try:
    import speech_recognition as sr
    from pydub import AudioSegment
    from gtts import gTTS
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    sr = None
    AudioSegment = None
    gTTS = None


class VoiceTools:
    """Voice-related tools for speech recognition and text-to-speech."""
    
    def __init__(self, system=None):
        """Initialize voice tools with optional system reference."""
        self.system = system
        self.logger = logger
        self.voice_enabled = settings.VOICE_ENABLED and VOICE_AVAILABLE
        
        if self.voice_enabled:
            self.recognizer = sr.Recognizer()
            self.logger.info("Voice tools initialized successfully")
        else:
            self.logger.warning("Voice tools not available - dependencies not installed or disabled")
    
    def speak(self, text: str) -> Dict[str, Any]:
        """
        Convert text to speech and play it.
        
        Args:
            text: The text to speak
            
        Returns:
            Dict with success status and message/error
        """
        if not self.voice_enabled:
            return {
                "success": False, 
                "error": "Voice mode not enabled or dependencies not available"
            }
        
        try:
            self.logger.info(f"Speaking: {text[:50]}...")
            
            # Create TTS object
            tts = gTTS(
                text=text, 
                lang=settings.TTS_LANGUAGE, 
                slow=False
            )
            
            # Save to temporary file
            temp_audio_file = settings.TEMP_DIR / "response.mp3"
            tts.save(str(temp_audio_file))
            
            # Play audio (try different players)
            success = False
            for player in ["mpg123", "mpv", "ffplay", "aplay"]:
                try:
                    subprocess.run(
                        [player, str(temp_audio_file)], 
                        check=True, 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL
                    )
                    success = True
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            # Clean up temporary file
            try:
                os.remove(temp_audio_file)
            except OSError:
                pass
            
            if success:
                return {"success": True, "output": f"Spoke: {text}"}
            else:
                return {"success": False, "error": "No audio player available"}
                
        except Exception as e:
            return {"success": False, "error": f"Failed to speak: {e}"}
    
    def listen(self, timeout: int = 5) -> Dict[str, Any]:
        """
        Listen for speech input and convert to text.
        
        Args:
            timeout: Timeout in seconds for listening
            
        Returns:
            Dict with success status and recognized text/error
        """
        if not self.voice_enabled:
            return {
                "success": False, 
                "error": "Voice mode not enabled or dependencies not available"
            }
        
        try:
            self.logger.info("Listening for speech...")
            
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=timeout)
                
                # Recognize speech
                self.logger.info("Recognizing speech...")
                text = self.recognizer.recognize_google(audio)
                
                self.logger.info(f"Recognized: {text}")
                return {"success": True, "output": text}
                
        except sr.UnknownValueError:
            return {"success": False, "error": "Could not understand audio"}
        except sr.RequestError as e:
            return {"success": False, "error": f"Speech recognition service error: {e}"}
        except sr.WaitTimeoutError:
            return {"success": False, "error": "Listening timeout - no speech detected"}
        except Exception as e:
            return {"success": False, "error": f"Error during speech recognition: {e}"}
    
    def convert_audio_format(self, input_file: str, output_file: str, 
                           output_format: str = "wav") -> Dict[str, Any]:
        """
        Convert audio file to different format.
        
        Args:
            input_file: Path to input audio file
            output_file: Path to output audio file
            output_format: Desired output format (wav, mp3, etc.)
            
        Returns:
            Dict with success status and message/error
        """
        if not VOICE_AVAILABLE:
            return {
                "success": False, 
                "error": "Audio processing dependencies not available"
            }
        
        try:
            self.logger.info(f"Converting {input_file} to {output_format}")
            
            # Load audio file
            audio = AudioSegment.from_file(input_file)
            
            # Export to new format
            audio.export(output_file, format=output_format)
            
            return {
                "success": True, 
                "output": f"Successfully converted to {output_file}"
            }
            
        except Exception as e:
            return {"success": False, "error": f"Failed to convert audio: {e}"}
    
    def get_voice_status(self) -> Dict[str, Any]:
        """
        Get voice tools status and capabilities.
        
        Returns:
            Dict with voice tools status information
        """
        return {
            "success": True,
            "output": {
                "voice_enabled": self.voice_enabled,
                "dependencies_available": VOICE_AVAILABLE,
                "settings_voice_enabled": settings.VOICE_ENABLED,
                "tts_language": settings.TTS_LANGUAGE,
                "tts_speed": settings.TTS_SPEED
            }
        }
    
    def test_voice_system(self) -> Dict[str, Any]:
        """
        Test the voice system functionality.
        
        Returns:
            Dict with test results
        """
        if not self.voice_enabled:
            return {
                "success": False,
                "error": "Voice system not enabled or dependencies not available"
            }
        
        try:
            # Test TTS
            test_text = "Voice system test successful"
            tts_result = self.speak(test_text)
            
            if not tts_result["success"]:
                return {
                    "success": False,
                    "error": f"TTS test failed: {tts_result['error']}"
                }
            
            return {
                "success": True,
                "output": "Voice system test completed successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Voice system test failed: {e}"
            }
