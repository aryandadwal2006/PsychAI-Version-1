import logging
from pathlib import Path
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    logging.warning("llama-cpp-python not installed, using fallback responses")

class PsychologyAssistant:
    def __init__(self):
        self.model_dir = Path("D:/PsychologyAI/models")
        
        if LLAMA_AVAILABLE:
            self._init_llama_model()
        else:
            self.model = None
            logging.info("Using fallback psychology responses")
            
        # Fallback responses for testing
        self.fallback_responses = [
            "I understand you're sharing something important with me. Can you tell me more about how that makes you feel?",
            "That sounds challenging. What thoughts go through your mind when this happens?",
            "Thank you for sharing that. How long has this been on your mind?",
            "I hear what you're saying. What would help you feel better about this situation?",
            "It takes courage to talk about difficult things. What support do you have in your life?"
        ]
        self.response_index = 0

    def _init_llama_model(self):
        """Initialize the Llama model if available"""
        try:
            model_files = list(self.model_dir.glob("*.gguf"))
            if not model_files:
                raise FileNotFoundError(f"No GGUF model found in {self.model_dir}")
                
            model_path = model_files[0]
            logging.info(f"Loading model: {model_path}")
            
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=2048,
                n_threads=4,
                n_gpu_layers=0,  # Use CPU only
                verbose=False
            )
            logging.info("Psychology AI model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self.model = None

    def generate_response(self, user_input, conversation_history=""):
        """Generate psychology response"""
        if not user_input.strip():
            return "I'm here to listen. What would you like to talk about?"
            
        if self.model and LLAMA_AVAILABLE:
            return self._generate_llama_response(user_input, conversation_history)
        else:
            return self._generate_fallback_response(user_input)

    def _generate_llama_response(self, user_input, history):
        """Generate response using Llama model"""
        try:
            system_prompt = """You are Dr. Mindwell, a compassionate AI psychologist. Respond with empathy, ask clarifying questions, and provide supportive guidance. Keep responses concise (2-3 sentences)."""
            
            prompt = f"{system_prompt}\n\nUser: {user_input}\nDr. Mindwell:"
            
            response = self.model.create_completion(
                prompt,
                max_tokens=100,
                temperature=0.7,
                stop=["User:", "\n\n"]
            )
            
            return response['choices'][0]['text'].strip()
            
        except Exception as e:
            logging.error(f"LLM generation error: {e}")
            return self._generate_fallback_response(user_input)

    def _generate_fallback_response(self, user_input):
        """Generate fallback response when model unavailable"""
        response = self.fallback_responses[self.response_index % len(self.fallback_responses)]
        self.response_index += 1
        return response
