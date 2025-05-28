import requests
import json
import time
import logging
import subprocess
import tempfile
from pathlib import Path
import soundfile as sf
import numpy as np

class NariLabsTTS:
    def __init__(self):
        self.base_dir = Path("D:/PsychologyAI")
        self.dia_dir = self.base_dir / "dia"
        self.audio_dir = self.base_dir / "audio_output"
        self.audio_dir.mkdir(exist_ok=True)
        
        # Try local Dia installation first, fallback to API
        self.use_local = self._check_local_dia()
        
        if self.use_local:
            logging.info("Using local Dia TTS installation")
            self._init_local_dia()
        else:
            logging.info("Using Nari Labs API endpoint")
            self.api_url = "https://api.deepinfra.com/v1/inference/nari-labs/Dia-1.6B"
            # Alternative: "https://nari-labs-dia.hf.space/api/predict"
        
    def _check_local_dia(self):
        """Check if local Dia installation exists"""
        dia_script = self.dia_dir / "app.py"
        return dia_script.exists()
    
    def _init_local_dia(self):
        """Initialize local Dia TTS"""
        try:
            # Import Dia locally
            import sys
            sys.path.append(str(self.dia_dir))
            from dia.model import Dia
            
            self.model = Dia.from_pretrained("nari-labs/Dia-1.6B")
            logging.info("Local Dia model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load local Dia: {e}")
            self.use_local = False

    def synthesize(self, text, speaker="Dr. Mindwell", emotion="calm", add_nonverbals=True):
        """Generate speech using Nari Labs Dia TTS"""
        if not text.strip():
            return None
            
        # Format text for Dia with psychology-appropriate styling
        formatted_text = self._format_psychology_text(text, speaker, emotion, add_nonverbals)
        
        if self.use_local:
            return self._synthesize_local(formatted_text)
        else:
            return self._synthesize_api(formatted_text)
    
    def _format_psychology_text(self, text, speaker, emotion, add_nonverbals):
        """Format text for psychology context with Dia speaker tags"""
        
        # Add appropriate non-verbal cues for psychology sessions
        if add_nonverbals:
            # Add thoughtful pauses and empathetic sounds
            if any(word in text.lower() for word in ['understand', 'i see', 'that makes sense']):
                text = text.replace('.', '. (thoughtful pause)')
            
            if any(word in text.lower() for word in ['difficult', 'challenging', 'hard']):
                text += ' (gentle sigh)'
            
            if any(word in text.lower() for word in ['good', 'excellent', 'progress']):
                text += ' (warm tone)'
        
        # Format with Dia speaker tags for consistent voice
        formatted = f"[S1] {text}"
        
        return formatted
    
    def _synthesize_local(self, text):
        """Generate audio using local Dia model"""
        try:
            output_audio = self.model.generate(
                text,
                temperature=0.8,  # Slightly more natural variation
                top_p=0.95,
                cfg_scale=3.5,    # Higher adherence to text
                max_new_tokens=2048
            )
            
            # Save audio file
            timestamp = int(time.time())
            output_path = self.audio_dir / f"nari_tts_{timestamp}.wav"
            sf.write(str(output_path), output_audio, 44100)
            
            logging.info(f"Nari TTS audio generated: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Local Dia TTS error: {e}")
            return None
    
    def _synthesize_api(self, text):
        """Generate audio using Nari Labs API"""
        payload = {
            "input": text,
            "temperature": 0.8,
            "top_p": 0.95,
            "cfg_scale": 3.5,
            "max_new_tokens": 2048,
            "speed": 1.0
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer YOUR_API_KEY"  # Get from DeepInfra
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=60
            )
            
            if response.status_code == 200:
                audio_data = response.json()["results"][0]["audio"]
                
                timestamp = int(time.time())
                output_path = self.audio_dir / f"nari_api_{timestamp}.wav"
                
                # Convert audio data to file
                audio_array = np.array(audio_data)
                sf.write(str(output_path), audio_array, 44100)
                
                logging.info(f"Nari API TTS generated: {output_path}")
                return str(output_path)
            else:
                logging.error(f"Nari API error: {response.status_code}")
                return None
                
        except Exception as e:
            logging.error(f"Nari API TTS error: {e}")
            return None
