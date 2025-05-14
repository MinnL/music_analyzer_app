import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import threading
import time
import base64
import io
import tempfile
import os
from scipy import signal

class AudioInput:
    def __init__(self, sample_rate=16000, channels=1, demo_mode_if_error=True, buffer_duration=60):
        self.sample_rate = sample_rate
        self.channels = channels
        self.demo_mode = False
        self.file_mode = False
        self.file_info = {}
        self.enable_demo_mode_if_error = demo_mode_if_error
        self.buffer_duration = buffer_duration  # Buffer duration in seconds
        
        # Audio buffer for real-time recording - more efficient circular buffer
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        self.audio_buffer = np.zeros(self.buffer_size)
        self.buffer_index = 0
        self.buffer_filled = False
        
        # Audio for playback
        self.current_audio = None
        self.current_audio_source = None
        
        # For threading
        self.recording = False
        self.record_thread = None
        self.lock = threading.Lock()
        
        # Cache for audio analysis
        self.audio_cache = {}
        self.max_cache_size = 5
        
        # Try to initialize audio device
        try:
            # List available devices to help with debugging
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            if len(input_devices) == 0:
                print("No input devices found. Will use demo mode.")
                self.demo_mode = True
            else:
                # Try to set a valid input device
                default_device = sd.query_devices(kind='input')
                print(f"Using input device: {default_device['name']}")
                
                # Test if we can open the input stream
                with sd.InputStream(samplerate=self.sample_rate, channels=self.channels):
                    pass  # Just testing if we can open the stream
                    
        except (sd.PortAudioError, Exception) as e:
            print(f"Error initializing audio device: {str(e)}")
            if self.enable_demo_mode_if_error:
                print("Switching to demo mode.")
                self.demo_mode = True
            else:
                raise
    
    def start_recording(self):
        """Start recording audio from microphone or generate demo audio."""
        # If already recording, stop first
        if self.recording:
            self.stop_recording()
        
        # Reset buffer
        self.audio_buffer = np.zeros(self.buffer_size)
        self.buffer_index = 0
        self.buffer_filled = False
        
        # Reset file mode
        self.file_mode = False
        self.file_info = {}
        
        # Start recording
        self.recording = True
        
        if self.demo_mode:
            self.record_thread = threading.Thread(target=self._generate_demo_audio)
        else:
            self.record_thread = threading.Thread(target=self._record_audio)
        
        self.record_thread.daemon = True
        self.record_thread.start()
    
    def stop_recording(self):
        """Stop recording audio."""
        # If not recording, nothing to do
        if not self.recording:
            return
            
        # Save current data before stopping
        audio_data = self.get_latest_data()
        if audio_data is not None and len(audio_data) > 0:
            print(f"Saving audio data before stopping: {len(audio_data)} samples")
            # Store it for later use
            self.current_audio = audio_data
            self.current_audio_source = "Recorded Audio" if not self.demo_mode else "Demo Audio"
            
        # Now stop the recording
        self.recording = False
        if self.record_thread:
            self.record_thread.join(timeout=1.0)
            self.record_thread = None
    
    def _record_audio(self):
        """Record audio from microphone."""
        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, callback=self._audio_callback):
                while self.recording:
                    time.sleep(0.1)  # Sleep to reduce CPU usage
        except (sd.PortAudioError, Exception) as e:
            print(f"Error recording audio: {str(e)}")
            if self.enable_demo_mode_if_error:
                print("Switching to demo mode.")
                self.demo_mode = True
                # Restart in demo mode
                self.record_thread = threading.Thread(target=self._generate_demo_audio)
                self.record_thread.daemon = True
                self.record_thread.start()
            else:
                self.recording = False
                raise
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream."""
        if status:
            print(f"Audio callback status: {str(status)}")
        
        # Process in smaller chunks for efficiency
        with self.lock:
            # Determine how many samples to copy
            samples_to_copy = min(len(indata), self.buffer_size - self.buffer_index)
            
            # Copy data to buffer
            if samples_to_copy > 0:
                self.audio_buffer[self.buffer_index:self.buffer_index + samples_to_copy] = indata[:samples_to_copy, 0]
                self.buffer_index += samples_to_copy
            
            # If we reached the end of the buffer, wrap around
            if self.buffer_index >= self.buffer_size:
                self.buffer_index = 0
                self.buffer_filled = True
                
                # If there's remaining data, copy it at the beginning
                remaining = len(indata) - samples_to_copy
                if remaining > 0:
                    self.audio_buffer[:remaining] = indata[samples_to_copy:, 0]
                    self.buffer_index = remaining
    
    def _generate_demo_audio(self):
        """Generate synthetic audio for demo mode."""
        # Simple synthetic audio generation with some variation
        buffer_duration_ms = 100  # Generate 100ms chunks for smoother updates
        buffer_samples = int(self.sample_rate * buffer_duration_ms / 1000)
        
        base_freq = 440  # Base frequency for synthesis
        
        t = 0  # Time counter
        while self.recording:
            # Generate synthetic audio chunk with varying frequency
            freq = base_freq + 50 * np.sin(t / 5)  # Slowly varying frequency
            times = np.linspace(t, t + buffer_duration_ms/1000, buffer_samples, endpoint=False)
            t += buffer_duration_ms/1000
            
            # Create a mixture of sine waves for more interesting audio
            audio_chunk = 0.5 * np.sin(2 * np.pi * freq * times)
            audio_chunk += 0.3 * np.sin(2 * np.pi * freq * 2 * times)  # First overtone
            audio_chunk += 0.2 * np.sin(2 * np.pi * freq * 3 * times)  # Second overtone
            
            # Add some noise
            audio_chunk += 0.05 * np.random.randn(len(audio_chunk))
            
            # Apply envelope for smoother transitions
            env = np.ones_like(audio_chunk)
            env[:min(100, len(env))] = np.linspace(0, 1, min(100, len(env)))
            env[-min(100, len(env)):] = np.linspace(1, 0, min(100, len(env)))
            audio_chunk *= env
            
            # Process in smaller chunks for efficiency
            with self.lock:
                # Determine how many samples to copy
                samples_to_copy = min(len(audio_chunk), self.buffer_size - self.buffer_index)
                
                # Copy data to buffer
                if samples_to_copy > 0:
                    self.audio_buffer[self.buffer_index:self.buffer_index + samples_to_copy] = audio_chunk[:samples_to_copy]
                    self.buffer_index += samples_to_copy
                
                # If we reached the end of the buffer, wrap around
                if self.buffer_index >= self.buffer_size:
                    self.buffer_index = 0
                    self.buffer_filled = True
                    
                    # If there's remaining data, copy it at the beginning
                    remaining = len(audio_chunk) - samples_to_copy
                    if remaining > 0:
                        self.audio_buffer[:remaining] = audio_chunk[samples_to_copy:]
                        self.buffer_index = remaining
            
            # Sleep to control generation rate
            time.sleep(buffer_duration_ms / 1000)
    
    def get_latest_data(self):
        """Get latest audio data for analysis."""
        with self.lock:
            # For file mode, return the processed file data
            if self.file_mode and hasattr(self, 'file_audio_data'):
                return self.file_audio_data
                
            # If recording is active, return buffer data    
            if self.recording:
                # For recording mode, return from circular buffer
                if self.buffer_filled:
                    # Return complete buffer with proper ordering
                    result = np.concatenate([
                        self.audio_buffer[self.buffer_index:],
                        self.audio_buffer[:self.buffer_index]
                    ])
                    return result
                else:
                    # Buffer not filled yet, return what we have
                    result = self.audio_buffer[:self.buffer_index]
                    return result
            
            # If not recording but we have saved audio, return that
            if self.current_audio is not None:
                print(f"Returning saved audio data: {len(self.current_audio)} samples")
                return self.current_audio
                
            print("Not recording and not in file mode, no audio data available")
            return None
    
    def process_uploaded_file(self, contents, filename):
        """Process an uploaded audio file."""
        try:
            # Decode the base64 content
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.' + filename.split('.')[-1], delete=False) as tmp:
                tmp.write(decoded)
                tmp_path = tmp.name
            
            # Load audio file with proper error handling
            try:
                audio_data, sr = librosa.load(tmp_path, sr=self.sample_rate, mono=True, res_type='kaiser_fast')
                
                # Trim to the first 20 seconds if longer (for efficiency)
                max_duration = 60  # seconds (increased from 30)
                max_samples = max_duration * self.sample_rate
                if len(audio_data) > max_samples:
                    audio_data = audio_data[:max_samples]
                
                # Set as current data
                self.file_mode = True
                self.file_info = {'filename': filename, 'path': tmp_path, 'duration': len(audio_data)/self.sample_rate}
                self.file_audio_data = audio_data
                
                # Also store for playback
                self.current_audio = audio_data
                self.current_audio_source = f"File: {filename}"
                
                return True, f"File loaded: {filename}"
            
            except Exception as e:
                os.unlink(tmp_path)
                print(f"Error loading audio file: {str(e)}")
                return False, f"Error loading audio file: {str(e)}"
                
        except Exception as e:
            print(f"Error processing uploaded file: {str(e)}")
            return False, f"Error processing file: {str(e)}"
    
    def save_current_audio(self):
        """Save the current audio for playback."""
        with self.lock:
            if self.file_mode:
                # File is already saved for playback
                print("In file mode, not saving audio for playback")
                return
            
            # Get latest data from buffer
            audio_data = self.get_latest_data()
            if audio_data is not None and len(audio_data) > 0:
                print(f"Saving audio data for playback: {len(audio_data)} samples")
                self.current_audio = audio_data
                self.current_audio_source = "Recorded Audio" if not self.demo_mode else "Demo Audio"
            else:
                print("No audio data available to save for playback")
    
    def get_audio_for_playback(self):
        """Get audio data for the player."""
        if self.current_audio is None:
            print("No current audio available for playback")
            return None
        
        try:
            # Convert to WAV format
            with io.BytesIO() as wav_io:
                sf.write(wav_io, self.current_audio, self.sample_rate, format='WAV')
                wav_bytes = wav_io.getvalue()
            
            # Create base64 encoded audio
            encoded = base64.b64encode(wav_bytes).decode('ascii')
            src = f"data:audio/wav;base64,{encoded}"
            
            print(f"Prepared audio for playback: {len(self.current_audio)/self.sample_rate:.2f} seconds")
            return {
                "data": src,
                "source": self.current_audio_source,
                "duration": len(self.current_audio) / self.sample_rate
            }
        except Exception as e:
            print(f"Error preparing audio for playback: {str(e)}")
            return None 