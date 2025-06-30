import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import requests
import av
import numpy as np
import io
import wave
import threading
import time
from collections import deque
import logging

# Configure logging to reduce noise
logging.getLogger("streamlit_webrtc").setLevel(logging.WARNING)

# Page configuration
st.set_page_config(
    page_title="Invoice Voice Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Invoice Voice Assistant")
st.markdown("Upload invoices and interact with them using voice commands")

# Configuration
BACKEND_URL = "http://localhost:8000"
CHAT_URL = "http://localhost:8000"
SAMPLE_RATE = 16000  # Common sample rate for speech recognition
CHUNK_DURATION = 3   # Process audio in 3-second chunks

# Initialize session state
def init_session_state():
    if 'invoices_uploaded' not in st.session_state:
        st.session_state.invoices_uploaded = []
    if 'voice_active' not in st.session_state:
        st.session_state.voice_active = False
    if 'audio_buffer' not in st.session_state:
        st.session_state.audio_buffer = deque(maxlen=SAMPLE_RATE * CHUNK_DURATION)
    if 'last_response_time' not in st.session_state:
        st.session_state.last_response_time = 0
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

init_session_state()

# Utility functions
def create_wav_bytes(audio_data, sample_rate=SAMPLE_RATE):
    """Convert audio data to WAV format bytes"""
    if len(audio_data) == 0:
        return None
        
    # Ensure audio is in the right format
    audio_array = np.array(audio_data, dtype=np.float32)
    
    # Normalize to 16-bit PCM
    audio_pcm = (audio_array * 32767).astype(np.int16)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  
        wav_file.setsampwidth(2)  
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_pcm.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer.getvalue()

def send_audio_to_backend(audio_bytes):
    """Send audio to chat backend and get response"""
    try:
        files = {"audio": ("audio.wav", audio_bytes, "audio/wav")}
        response = requests.post(f"{CHAT_URL}/chat", files=files, timeout=15)
        
        if response.status_code == 200:
            return response.content, True
        else:
            st.error(f"Chat service error: {response.status_code}")
            return None, False
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None, False
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None, False

# Audio processing class
class AudioProcessor:
    def __init__(self):
        self.audio_buffer = deque(maxlen=SAMPLE_RATE * 10)  
        self.lock = threading.Lock()
        self.frames_received = 0
        self.last_process_time = 0
        
    def process_audio_frame(self, frame: av.AudioFrame):
        """Process incoming audio frame"""
        try:
            # Convert frame to numpy array
            audio_array = frame.to_ndarray().flatten()
            with self.lock:
                # Add to buffer
                self.audio_buffer.extend(audio_array)
                self.frames_received += 1
                
                # Process if we have enough data and enough time has passed
                current_time = time.time()
                if (len(self.audio_buffer) >= SAMPLE_RATE * CHUNK_DURATION and 
                    current_time - self.last_process_time > CHUNK_DURATION):
                    
                    self.last_process_time = current_time
                    return self._process_chunk()
                    
        except Exception as e:
            st.error(f"Audio processing error: {e}")
            
        return frame
        
    def _process_chunk(self):
        """Process accumulated audio chunk"""
        try:
            # Get audio data
            audio_data = list(self.audio_buffer)
            self.audio_buffer.clear()
            
            # Convert to WAV
            wav_bytes = create_wav_bytes(audio_data)
            if wav_bytes:
                # Send to backend in a separate thread to avoid blocking
                threading.Thread(
                    target=self._send_audio_async, 
                    args=(wav_bytes,), 
                    daemon=True
                ).start()
                
        except Exception as e:
            st.error(f"Chunk processing error: {e}")
            
    def _send_audio_async(self, audio_bytes):
        """Send audio to backend asynchronously"""
        response_audio, success = send_audio_to_backend(audio_bytes)
        
        if success and response_audio:
            # Store response in session state for playback
            st.session_state.last_response = response_audio
            st.session_state.last_response_time = time.time()
            
    def get_stats(self):
        """Get processing statistics"""
        with self.lock:
            return {
                'frames_received': self.frames_received,
                'buffer_size': len(self.audio_buffer),
                'buffer_seconds': len(self.audio_buffer) / SAMPLE_RATE if self.audio_buffer else 0
            }

# Create audio processor
audio_processor = AudioProcessor()

# Sidebar - Configuration and Status
with st.sidebar:
    st.header("⚙️ Configuration")
    # Audio stats
    if st.session_state.voice_active:
        stats = audio_processor.get_stats()
        st.write(f"Frames: {stats['frames_received']}")
        st.write(f"Buffer: {stats['buffer_seconds']:.1f}s")

# Main content area
col1, col2 = st.columns([1, 1])

# Column 1 - Invoice Upload
with col1:
    st.header("📄 Invoice Management")
    
    uploaded_files = st.file_uploader(
        "Upload invoices (PDF, PNG, JPG)",
        type=["pdf", "png", "jpg"],
        accept_multiple_files=True,
        help="You can upload 1-4 invoice files"
    )
    
    if st.button("📤 Process Invoices", disabled=not uploaded_files):
        if uploaded_files:
            with st.spinner("Processing invoices..."):
                try:
                    # Prepare files for upload
                    files = []
                    for uploaded_file in uploaded_files:
                        files.append(("files", (uploaded_file.name, uploaded_file.getvalue())))
                    
                    # Send to backend
                    response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        uploaded_invoices = [item["invoice_id"] for item in result.get("uploaded", [])]
                        st.session_state.invoices_uploaded.extend(uploaded_invoices)
                        
                        st.success(f"✅ Processed {len(uploaded_invoices)} invoices:")
                        for invoice_id in uploaded_invoices:
                            st.write(f"  • {invoice_id}")
                    else:
                        st.error(f"❌ Upload failed: HTTP {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Upload error: {str(e)}")
    
    # Show uploaded invoices
    if st.session_state.invoices_uploaded:
        st.subheader("📋 Uploaded Invoices")
        for invoice_id in st.session_state.invoices_uploaded:
            st.write(f"✓ {invoice_id}")
        
        if st.button("🗑️ Clear All Invoices"):
            st.session_state.invoices_uploaded.clear()
            st.rerun()

# Column 2 - Voice Interface
with col2:
    st.header("🎤 Voice Interface")
    
    # Voice control buttons
    button_col1, button_col2 = st.columns(2)
    
    with button_col1:
        if st.button("🎙️ Start Voice Chat", disabled=st.session_state.voice_active):
            st.session_state.voice_active = True
            st.rerun()
    
    with button_col2:
        if st.button("🛑 Stop Voice Chat", disabled=not st.session_state.voice_active):
            st.session_state.voice_active = False
            st.rerun()
    
    # Voice chat interface
    if st.session_state.voice_active:
        st.info("🎤 Voice chat is active - Start speaking!")
        
        # WebRTC Configuration
        rtc_config = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]}
            ]
        })
        
        # Create WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="voice_chat",
            mode=WebRtcMode.SENDONLY,
            rtc_configuration=rtc_config,
            media_stream_constraints={
                "audio": {
                    "sampleRate": SAMPLE_RATE,
                    "channelCount": 1,
                    "echoCancellation": True,
                    "noiseSuppression": True,
                    "autoGainControl": True,
                },
                "video": False
            },
            audio_frame_callback=audio_processor.process_audio_frame,
            async_processing=True,
        )
        
        # Show connection status
        if webrtc_ctx.state.playing:
            st.success("🔗 Connected - Listening for speech...")
        elif webrtc_ctx.state.signalling:
            st.info("🔄 Connecting...")
        else:
            st.warning("⚠️ Not connected - Check microphone permissions")
        
        # Play response if available
        if (hasattr(st.session_state, 'last_response') and 
            st.session_state.last_response and
            time.time() - st.session_state.last_response_time < 2):  # Play within 2 seconds
            
            st.success("🤖 AI Response:")
            st.audio(st.session_state.last_response, format="audio/mp3")
            
            # Add to conversation history
            st.session_state.conversation_history.append({
                "timestamp": time.strftime("%H:%M:%S"),
                "type": "ai_response"
            })
            
            # Clear the response to avoid replaying
            st.session_state.last_response = None
