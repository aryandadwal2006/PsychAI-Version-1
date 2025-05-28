import gradio as gr
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our modules
try:
    from psychology_ai.core.stt_manager import WhisperSTT
    from psychology_ai.core.llm_manager import PsychologyAssistant
    from psychology_ai.core.tts_manager import SimpleTTS
except ImportError as e:
    logger.error(f"Import error: {e}")
    raise

# Initialize components
logger.info("Initializing Psychology AI components...")
try:
    stt_engine = WhisperSTT()
    psychologist = PsychologyAssistant()
    tts_engine = SimpleTTS()
    logger.info("All components initialized successfully!")
except Exception as e:
    logger.error(f"Initialization failed: {e}")
    raise

# Conversation history
conversation_history = []

def process_audio(audio_path):
    """Process recorded audio through the full pipeline"""
    if not audio_path:
        return [], None, "No audio recorded"
        
    try:
        logger.info(f"Processing audio: {audio_path}")
        
        # Step 1: Speech-to-Text
        user_text = stt_engine.transcribe(audio_path)
        if not user_text:
            return conversation_history, None, "Could not transcribe audio. Please try again."
            
        logger.info(f"Transcribed: {user_text}")
        conversation_history.append(["You", user_text])
        
        # Step 2: Generate AI Response
        ai_response = psychologist.generate_response(user_text, "\n".join([f"{role}: {text}" for role, text in conversation_history]))
        logger.info(f"AI Response: {ai_response}")
        conversation_history.append(["Dr. Mindwell", ai_response])
        
        # Step 3: Text-to-Speech
        tts_path = tts_engine.synthesize(ai_response)
        
        return conversation_history, tts_path, ""
        
    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        logger.error(error_msg)
        return conversation_history, None, error_msg

def clear_conversation():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return [], None, ""

# Gradio Interface
with gr.Blocks(css=".gradio-container {max-width: 900px !important}") as app:
    gr.Markdown("# Psychology AI Assistant 🤖💡")
    gr.Markdown("**Speak your thoughts and receive supportive guidance**")
    
    with gr.Row():
        with gr.Column(scale=2):
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 Speak your thoughts",
                show_label=True
            )
            
        with gr.Column(scale=1):
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
            
    with gr.Row():
        audio_output = gr.Audio(
            autoplay=True,
            visible=True,
            label="🔊 AI Response Audio"
        )
        
    chat_history = gr.Chatbot(
        height=400,
        label="💬 Conversation"
    )
    
    error_output = gr.Textbox(
        label="Status",
        visible=True,
        interactive=False
    )
    
    # Event handlers
    audio_input.stop_recording(
        process_audio,
        inputs=[audio_input],
        outputs=[chat_history, audio_output, error_output]
    )
    
    clear_btn.click(
        clear_conversation,
        outputs=[chat_history, audio_output, error_output]
    )

if __name__ == "__main__":
    logger.info("Starting Psychology AI Assistant...")
    app.launch(
        server_port=7860,
        server_name="127.0.0.1",
        show_error=True,
        debug=True
    )
