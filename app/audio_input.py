import pyaudio
import numpy as np
import threading
import time
import queue

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
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.thread = None
        
        # Buffer to store audio chunks
        self.buffer = queue.Queue(maxsize=buffer_size)
        
    def start_recording(self):
        """Start recording audio from microphone"""
        if self.is_recording:
            return
        
        # Clear buffer
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                break
        
        # Open audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        self.is_recording = True
        
    def stop_recording(self):
        """Stop recording audio"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
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
        self.audio.terminate() 