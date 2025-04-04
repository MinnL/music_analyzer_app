import pyaudio
import numpy as np
import threading
import time
import queue
import os
import sys
import warnings
import librosa
import io
import base64

class AudioInput:
    """
    Class for capturing real-time audio input from microphone
    """
    
    def __init__(self, 
                 sample_rate=22050, 
                 chunk_size=1024, 
                 channels=1, 
                 format=pyaudio.paFloat32,
                 buffer_size=10):
        """
        Initialize the audio input module
        
        Args:
            sample_rate: Sample rate in Hz
            chunk_size: Number of frames per buffer
            channels: Number of channels (1 for mono, 2 for stereo)
            format: Audio format (from pyaudio constants)
            buffer_size: Maximum number of chunks to keep in buffer
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format
        
        # Initialize PyAudio
        try:
            self.audio = pyaudio.PyAudio()
            self.available_devices = self._get_available_devices()
        except Exception as e:
            print(f"Warning: Error initializing PyAudio: {e}")
            self.audio = None
            self.available_devices = []
        
        self.stream = None
        self.is_recording = False
        self.thread = None
        self.demo_mode = False
        self.file_mode = False
        self.file_audio_data = None
        self.file_info = None
        
        # Buffer to store audio chunks
        self.buffer = queue.Queue(maxsize=buffer_size)
        
    def _get_available_devices(self):
        """Get a list of available input devices"""
        devices = []
        if self.audio:
            info = self.audio.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            
            for i in range(num_devices):
                device_info = self.audio.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxInputChannels') > 0:
                    devices.append({
                        'index': i,
                        'name': device_info.get('name'),
                        'channels': device_info.get('maxInputChannels')
                    })
                    
        return devices
        
    def start_recording(self):
        """Start recording audio from microphone"""
        # Reset file mode if it was active
        self.file_mode = False
        self.file_audio_data = None
        self.file_info = None
        
        if self.is_recording:
            return
        
        # Clear buffer
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                break
                
        # If PyAudio is not available, use demo mode
        if not self.audio or not self.available_devices:
            print("No audio devices available. Starting in demo mode.")
            self.demo_mode = True
            self.is_recording = True
            
            # Start a thread to simulate audio input
            self.thread = threading.Thread(target=self._generate_demo_audio)
            self.thread.daemon = True
            self.thread.start()
            return
        
        # Try each available input device until one works
        for device in self.available_devices:
            try:
                print(f"Trying to open audio device: {device['name']}")
                # Open audio stream
                self.stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=device['index'],
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_callback
                )
                
                self.is_recording = True
                print(f"Successfully opened audio device: {device['name']}")
                return
            except Exception as e:
                print(f"Failed to open device {device['name']}: {e}")
                if self.stream:
                    self.stream.close()
                    self.stream = None
                    
        # If all devices failed, use demo mode
        print("All audio devices failed. Starting in demo mode.")
        self.demo_mode = True
        self.is_recording = True
        
        # Start a thread to simulate audio input
        self.thread = threading.Thread(target=self._generate_demo_audio)
        self.thread.daemon = True
        self.thread.start()
    
    def process_uploaded_file(self, contents, filename):
        """
        Process uploaded audio file
        
        Args:
            contents: File contents as base64 string
            filename: Name of the uploaded file
            
        Returns:
            Tuple of (success, message)
        """
        # Stop any active recording
        if self.is_recording:
            self.stop_recording()
        
        # Reset file mode
        self.file_mode = False
        self.file_audio_data = None
        self.file_info = None
        
        try:
            # Decode the base64 string
            content_type, content_string = contents.split(',')
            decoded = io.BytesIO(base64.b64decode(content_string))
            
            # Check file extension
            ext = os.path.splitext(filename)[1].lower()
            supported_formats = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
            
            if ext not in supported_formats:
                return False, f"Unsupported file format: {ext}. Please upload a WAV, MP3, OGG, FLAC, or M4A file."
            
            # Load the audio file with librosa
            audio_data, sr = librosa.load(decoded, sr=self.sample_rate, mono=True)
            
            # Store the audio data
            self.file_audio_data = audio_data
            self.file_mode = True
            self.file_info = {
                'filename': filename,
                'duration': len(audio_data) / self.sample_rate,
                'sample_rate': self.sample_rate
            }
            
            return True, f"Successfully loaded audio file: {filename} ({self.file_info['duration']:.2f} seconds)"
            
        except Exception as e:
            print(f"Error processing uploaded file: {e}")
            return False, f"Error processing file: {str(e)}"
        
    def _generate_demo_audio(self):
        """Generate sample audio data for demo mode"""
        while self.is_recording:
            # Generate a simple sine wave as a demo
            t = np.arange(0, self.chunk_size) / self.sample_rate
            
            # Create a mix of frequencies to simulate music
            frequencies = [440, 880, 1320]  # A4, A5, E6
            audio_data = np.zeros(self.chunk_size, dtype=np.float32)
            
            for freq in frequencies:
                audio_data += 0.2 * np.sin(2 * np.pi * freq * t)
                
            # Add some noise
            audio_data += 0.05 * np.random.randn(self.chunk_size).astype(np.float32)
            
            # Normalize
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Put data in buffer, discard if buffer is full
            try:
                self.buffer.put_nowait(audio_data)
            except queue.Full:
                try:
                    self.buffer.get_nowait()
                    self.buffer.put_nowait(audio_data)
                except queue.Empty:
                    pass
                    
            # Sleep to simulate real-time
            time.sleep(self.chunk_size / self.sample_rate)
        
    def stop_recording(self):
        """Stop recording audio"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        self.demo_mode = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Put data in buffer, discard if buffer is full
        try:
            self.buffer.put_nowait(audio_data)
        except queue.Full:
            # If buffer is full, remove oldest chunk and add new one
            try:
                self.buffer.get_nowait()
                self.buffer.put_nowait(audio_data)
            except queue.Empty:
                pass
            
        return (in_data, pyaudio.paContinue)
    
    def get_latest_data(self, seconds=3):
        """
        Get the latest audio data
        
        Args:
            seconds: Number of seconds of audio to return
            
        Returns:
            Numpy array of audio data
        """
        # If in file mode, return the file audio data
        if self.file_mode and self.file_audio_data is not None:
            return self.file_audio_data
            
        if not self.is_recording:
            return np.array([])
        
        # Calculate how many chunks we need for the requested duration
        chunks_needed = int(seconds * self.sample_rate / self.chunk_size)
        
        # Get all available chunks
        chunks = []
        try:
            for _ in range(self.buffer.qsize()):
                chunks.append(self.buffer.get())
        except queue.Empty:
            pass
        
        # Put chunks back in buffer
        for chunk in chunks:
            try:
                self.buffer.put_nowait(chunk)
            except queue.Full:
                break
                
        # Take only the most recent chunks needed
        if len(chunks) > chunks_needed:
            chunks = chunks[-chunks_needed:]
            
        if not chunks:
            return np.array([])
            
        # Concatenate chunks
        return np.concatenate(chunks)
        
    def __del__(self):
        """Clean up resources when object is destroyed"""
        self.stop_recording()
        if self.audio:
            self.audio.terminate() 