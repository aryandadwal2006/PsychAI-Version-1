import logging
import time
from pathlib import Path
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("pyttsx3 not installed, TTS unavailable")

class SimpleTTS:
    def __init__(self):
        self.audio_dir = Path("D:/PsychologyAI/audio_output")
        self.audio_dir.mkdir(exist_ok=True)
        
        if TTS_AVAILABLE:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 180)  # Speed
                self.engine.setProperty('volume', 0.9)  # Volume
                logging.info("TTS engine initialized")
            except Exception as e:
                logging.error(f"TTS initialization failed: {e}")
                self.engine = None
        else:
            self.engine = None

    def synthesize(self, text):
        """Convert text to speech and return audio file path"""
        if not text.strip():
            return None
            
        if not self.engine:
            logging.warning("TTS not available")
            return None
            
        try:
            output_path = self.audio_dir / f"tts_{int(time.time())}.wav"
            self.engine.save_to_file(text, str(output_path))
            self.engine.runAndWait()
            
            if output_path.exists():
                logging.info(f"TTS audio saved: {output_path}")
                return str(output_path)
            else:
                logging.error("TTS file not created")
                return None
                
        except Exception as e:
            logging.error(f"TTS Error: {e}")
            return None
