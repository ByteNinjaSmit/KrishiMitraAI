# Voice Assistant for Krishi Mitra
# Provides speech-to-text and text-to-speech capabilities for better accessibility

import streamlit as st
from utils import detect_language
from gtts import gTTS

try:
    import speech_recognition as sr
    import pyttsx3
    import tempfile
    import wave
    import threading
    from io import BytesIO
    import base64
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

class VoiceAssistant:
    def __init__(self):
        """Initialize voice assistant with speech recognition and synthesis"""
        self.voice_available = VOICE_AVAILABLE
        self.recognizer = None
        self.tts_engine = None
        
        if self.voice_available:
            try:
                self.recognizer = sr.Recognizer()
                # Initialize TTS engine in a separate thread to avoid blocking
                self.init_tts()
            except Exception as e:
                st.warning(f"Voice features partially unavailable: {str(e)}")
                self.voice_available = False

    def generate_malayalam_audio(self, text: str) -> str:
        """Generate Malayalam audio using gTTS and return HTML player"""
        try:
            from io import BytesIO
            import base64
            tts = gTTS(text=text, lang="ml")
            audio_fp = BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            audio_base64 = base64.b64encode(audio_fp.read()).decode()
            return f"""
            <audio controls autoplay style="margin-top:10px; width:100%;">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            """
        except Exception as e:
            return f"<p style='color:red;'>Malayalam TTS failed: {str(e)}</p>"


    
    def init_tts(self):
        """Initialize text-to-speech engine"""
        try:
            self.tts_engine = pyttsx3.init()
            # Set properties
            self.tts_engine.setProperty('rate', 150)  # Speed of speech
            self.tts_engine.setProperty('volume', 0.8)  # Volume (0.0 to 1.0)
            
            # Try to set voice to a more natural one
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
        except Exception as e:
            st.warning(f"Text-to-speech not available: {str(e)}")
            self.tts_engine = None
    
    def create_voice_recorder_html(self) -> str:
        """Create HTML/JavaScript for voice recording using Web Speech API"""
        return """
        <div id="voice-container" style="text-align: center; padding: 20px;">
            <button id="voice-btn" onclick="toggleRecording()" 
                    style="background: linear-gradient(135deg, #4CAF50, #45a049); 
                           color: white; border: none; padding: 15px 30px; 
                           border-radius: 25px; font-size: 16px; cursor: pointer;
                           box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);">
                🎤 Click to Speak (Malayalam/English)
            </button>
            <button id="voice-stop" onclick="forceStop()" 
                    style="background: #f44336; color: white; border: none; padding: 12px 22px; 
                           border-radius: 25px; font-size: 14px; cursor: pointer; margin-left: 10px;">
                ⏹️ Stop
            </button>
            <div id="voice-status" style="margin-top: 10px; font-weight: bold;"></div>
            <div id="voice-result" style="margin-top: 15px; padding: 10px; 
                                          background: #f0f8ff; border-radius: 10px; 
                                          min-height: 50px; display: none;"></div>
        </div>

        <script>
        let recognition;
        let isRecording = false;

        // Check if speech recognition is supported
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            
            // Configure recognition
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;
            
            // Prefer Malayalam; we'll auto-switch to English based on detected characters
            recognition.lang = 'ml-IN';
            
            recognition.onstart = function() {
                document.getElementById('voice-status').textContent = '🎤 Listening... Speak now!';
                document.getElementById('voice-btn').textContent = '🔴 Recording... Click to Stop';
                document.getElementById('voice-btn').style.background = 'linear-gradient(135deg, #f44336, #d32f2f)';
            };
            
            recognition.onresult = function(event) {
                const result = event.results[0][0].transcript;
                document.getElementById('voice-result').textContent = 'You said: ' + result;
                document.getElementById('voice-result').style.display = 'block';
                
                // Send result to Streamlit
                window.parent.postMessage({
                    type: 'voice_input',
                    text: result
                }, '*');
                
                // Auto-submit the question
                const inputElement = window.parent.document.querySelector('textarea[placeholder*="Ask"]');
                if (inputElement) {
                    inputElement.value = result;
                    inputElement.dispatchEvent(new Event('input', { bubbles: true }));
                }
            };
            
            recognition.onerror = function(event) {
                document.getElementById('voice-status').textContent = '❌ Error: ' + event.error;
                resetButton();
            };
            
            recognition.onend = function() {
                resetButton();
            };
        } else {
            document.getElementById('voice-btn').textContent = '❌ Voice not supported in this browser';
            document.getElementById('voice-btn').disabled = true;
            document.getElementById('voice-stop').disabled = true;
        }

        function toggleRecording() {
            if (!recognition) return;
            
            if (isRecording) {
                recognition.stop();
            } else {
                // Try Malayalam first, then fallback to English
                recognition.lang = 'ml-IN';
                recognition.start();
                isRecording = true;
            }
        }

        function forceStop() {
            if (!recognition) return;
            try { recognition.stop(); } catch (e) {}
            resetButton();
        }

        function resetButton() {
            isRecording = false;
            document.getElementById('voice-status').textContent = '';
            document.getElementById('voice-btn').textContent = '🎤 Click to Speak (Malayalam/English)';
            document.getElementById('voice-btn').style.background = 'linear-gradient(135deg, #4CAF50, #45a049)';
        }

        // Try English if Malayalam fails
        if (recognition) {
            recognition.onerror = function(event) {
                if (event.error === 'language-not-supported' && recognition.lang === 'ml-IN') {
                    recognition.lang = 'en-IN';
                    recognition.start();
                    document.getElementById('voice-status').textContent = '🎤 Trying English... Speak now!';
                } else {
                    document.getElementById('voice-status').textContent = '❌ Error: ' + event.error;
                    resetButton();
                }
            };
            // Quick heuristic: switch to English if transcript looks ASCII/Latin
            recognition.onresult = function(event) {
                let result = event.results[0][0].transcript;
                if (/^[\x00-\x7F]+$/.test(result) && recognition.lang !== 'en-IN') {
                    // Likely English; restart quickly in English to improve accuracy
                    try { recognition.stop(); } catch (e) {}
                    recognition.lang = 'en-IN';
                    recognition.start();
                    document.getElementById('voice-status').textContent = '🎤 Switching to English...';
                    return;
                }
                document.getElementById('voice-result').textContent = 'You said: ' + result;
                document.getElementById('voice-result').style.display = 'block';
                window.parent.postMessage({ type: 'voice_input', text: result }, '*');
                const inputElement = window.parent.document.querySelector('textarea[placeholder*="Ask"]');
                if (inputElement) {
                    inputElement.value = result;
                    inputElement.dispatchEvent(new Event('input', { bubbles: true }));
                }
            };
        }
        </script>
        """
    
    def create_audio_player_html(self, text: str, lang_code: str) -> str:
        """Create HTML for audio playback. Uses gTTS for Malayalam, Web Speech API for English."""
        # Clean text for speech
        clean_text = text.replace('*', '').replace('#', '').replace('`', '')
        clean_text = clean_text.replace('\n', '. ')
        
        # Limit text length for better performance
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "..."
        
        # ✅ Malayalam handled via gTTS
        if lang_code.startswith("ml"):
            return self.generate_malayalam_audio(clean_text)

        # ✅ English handled via Web Speech API
        return f"""
        <div style="background: linear-gradient(135deg, #2196F3, #1976D2); 
                    padding: 15px; border-radius: 10px; margin: 10px 0;">
            <button onclick="speakText()" 
                    style="background: white; color: #1976D2; border: none; 
                           padding: 10px 20px; border-radius: 20px; 
                           font-size: 14px; cursor: pointer; font-weight: bold;">
                🔊 Listen to Response
            </button>
            <button onclick="stopSpeaking()" 
                    style="background: #f44336; color: white; border: none; 
                           padding: 10px 20px; border-radius: 20px; 
                           font-size: 14px; cursor: pointer; margin-left: 10px;">
                ⏹️ Stop
            </button>
            <div id="speech-status" style="color: white; margin-top: 10px; font-size: 12px;"></div>
        </div>

        <script>
        let speechSynthesis = window.speechSynthesis;
        let currentUtterance = null;

        function speakText() {{
            if (speechSynthesis.speaking) {{
                speechSynthesis.cancel();
            }}
            
            const text = `{clean_text}`;
            currentUtterance = new SpeechSynthesisUtterance(text);
            currentUtterance.lang = '{lang_code}';
            currentUtterance.rate = 0.9;
            currentUtterance.pitch = 1;
            currentUtterance.volume = 0.8;
            
            const voices = speechSynthesis.getVoices();
            let selected = null;
            for (let v of voices) {{
                if (v.lang && (v.lang.includes('en-IN') || v.lang.includes('en-GB') || v.lang.includes('en-US'))) {{
                    selected = v; break;
                }}
            }}
            if (selected) {{ currentUtterance.voice = selected; }}
            ~
            currentUtterance.onstart = function() {{
                document.getElementById('speech-status').textContent = '🔊 Speaking...';
            }};
            
            currentUtterance.onend = function() {{
                document.getElementById('speech-status').textContent = '✅ Finished speaking';
            }};
            
            currentUtterance.onerror = function(event) {{
                document.getElementById('speech-status').textContent = '❌ Speech error: ' + event.error;
            }};
            
            speechSynthesis.speak(currentUtterance);
        }}

        function stopSpeaking() {{
            if (speechSynthesis.speaking) {{
                speechSynthesis.cancel();
                document.getElementById('speech-status').textContent = '⏹️ Stopped';
            }}
        }}
        </script>
        """

    
    def display_voice_input_interface(self):
        """Display voice input interface in Streamlit"""
        st.markdown("### 🎤 Voice Input")
        st.markdown("**Speak your question in Malayalam or English**")
        
        # Display voice recorder
        st.components.v1.html(self.create_voice_recorder_html(), height=200)
        
        # Instructions
        with st.expander("📝 Voice Input Instructions"):
            st.markdown("""
            **How to use voice input:**
            1. Click the microphone button
            2. Wait for "Listening..." message
            3. Speak your question clearly in Malayalam or English
            4. The system will automatically convert speech to text
            5. Your question will appear in the chat input box
            
            **Tips for better recognition:**
            - Speak clearly and at moderate speed
            - Ensure minimal background noise
            - Use simple, direct questions
            - Wait for the recording indicator before speaking
            
            **Language Support:**
            - Malayalam (മലയാളം) - Primary
            - English - Fallback
            """)
    
    def create_voice_output_for_response(self, response_text: str):
        """Create voice output interface for AI response"""
        if not response_text or len(response_text.strip()) < 10:
            return
        
        # Detect language for TTS selection
        try:
            lang = detect_language(response_text)
        except Exception:
            lang = "english"
        lang_code = 'ml-IN' if lang == 'malayalam' else 'en-IN'
        st.markdown("### 🔊 Audio Response")
        st.components.v1.html(self.create_audio_player_html(response_text, lang_code), height=120)
    
    def get_voice_input_javascript(self) -> str:
        """Get JavaScript code for voice input handling"""
        return """
        <script>
        // Handle voice input from iframe
        window.addEventListener('message', function(event) {
            if (event.data.type === 'voice_input') {
                const textArea = document.querySelector('textarea[placeholder*="Ask"]');
                if (textArea) {
                    textArea.value = event.data.text;
                    textArea.dispatchEvent(new Event('input', { bubbles: true }));
                    textArea.focus();
                }
            }
        });
        </script>
        """
    
    def is_voice_available(self) -> bool:
        """Check if voice features are available"""
        return True  # Web Speech API is available in modern browsers
    
    def get_voice_setup_instructions(self) -> str:
        """Get instructions for voice setup"""
        return """
        **Voice Features Setup:**
        
        Voice input and output are built into modern web browsers using the Web Speech API.
        
        **Requirements:**
        - Modern web browser (Chrome, Firefox, Safari, Edge)
        - Microphone access (will prompt for permission)
        - Internet connection for speech processing
        
        **Supported Languages:**
        - Malayalam (മലയാളം) - Primary
        - English (Indian) - Fallback
        
        **Browser Compatibility:**
        ✅ Google Chrome (Recommended)
        ✅ Microsoft Edge
        ✅ Safari (iOS/macOS)
        ⚠️ Firefox (Limited support)
        
        **For best results:**
        - Use Chrome or Edge browsers
        - Allow microphone permissions when prompted
        - Ensure stable internet connection
        - Speak in a quiet environment
        """

# Helper functions for Streamlit integration
def initialize_voice_assistant():
    """Initialize voice assistant in Streamlit session state"""
    if 'voice_assistant' not in st.session_state:
        st.session_state.voice_assistant = VoiceAssistant()
    return st.session_state.voice_assistant

def add_voice_capabilities_to_chat():
    """Add voice input/output to chat interface"""
    voice_assistant = initialize_voice_assistant()
    
    # Add voice input interface
    with st.expander("🎤 Voice Input", expanded=False):
        voice_assistant.display_voice_input_interface()
    
    return voice_assistant