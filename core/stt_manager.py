import subprocess
import tempfile
import logging
from pathlib import Path
from pydub import AudioSegment

class WhisperSTT:
    def __init__(self):
        self.base_dir = Path("D:/PsychologyAI")
        self.whisper_exe = self.base_dir / "whisper.cpp" / "build" / "bin" / "Release" / "whisper-cli.exe"
        self.model_path = self.base_dir / "whisper_cpp" / "models" / "ggml-base.en.bin"
        self.temp_dir = self.base_dir / "temp_audio"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Verify components exist
        if not self.whisper_exe.exists():
            raise FileNotFoundError(f"Whisper executable not found: {self.whisper_exe}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Whisper model not found: {self.model_path}")
            
        logging.info(f"WhisperSTT initialized: {self.whisper_exe}")

    def transcribe(self, audio_path):
        """Transcribe audio file to text"""
        if not audio_path or not Path(audio_path).exists():
            logging.error(f"Audio file not found: {audio_path}")
            return ""
            
        try:
            # Convert to WAV format if needed
            wav_path = self._ensure_wav_format(audio_path)
            
            # Run whisper-cli
            cmd = [
                str(self.whisper_exe),
                "-m", str(self.model_path),
                "-f", str(wav_path),
                "--no-timestamps",
                "--threads", "4"
            ]
            
            logging.info(f"Running whisper command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                cwd=str(self.whisper_exe.parent),
                timeout=30
            )
            
            if result.returncode == 0:
                text = self._parse_whisper_output(result.stdout)
                logging.info(f"Transcription successful: {text[:50]}...")
                return text
            else:
                logging.error(f"Whisper error: {result.stderr}")
                return ""
                
        except Exception as e:
            logging.error(f"STT Error: {str(e)}")
            return ""

    def _ensure_wav_format(self, audio_path):
        """Convert audio to WAV format if needed"""
        audio_path = Path(audio_path)
        if audio_path.suffix.lower() == '.wav':
            return audio_path
            
        # Convert to WAV
        try:
            audio = AudioSegment.from_file(str(audio_path))
            wav_path = self.temp_dir / f"converted_{audio_path.stem}.wav"
            audio.export(str(wav_path), format="wav", parameters=["-ar", "16000", "-ac", "1"])
            return wav_path
        except Exception as e:
            logging.error(f"Audio conversion error: {e}")
            return audio_path

    def _parse_whisper_output(self, output):
        """Extract text from whisper output"""
        lines = output.strip().split('\n')
        text_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('[') and not line.startswith('whisper_'):
                text_lines.append(line)
        return ' '.join(text_lines).strip()
