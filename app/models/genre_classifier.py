import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import os
import json
import time
import librosa
import hashlib

# GTZAN genres list
GTZAN_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
               'jazz', 'metal', 'pop', 'reggae', 'rock']

class GTZAN_CNN(nn.Module):
    """
    CNN model for GTZAN genre classification
    """
    def __init__(self):
        super(GTZAN_CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=4)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 genres
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Input shape: [batch, 1, height, width]
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = self.global_pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class GenreClassifier:
    """
    Wrapper for genre classification model
    """
    def __init__(self, model_path=None, use_cuda=torch.cuda.is_available()):
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = GTZAN_CNN().to(self.device)
        
        # These are the genre classes for the GTZAN dataset
        self.genres = [
            'blues', 'classical', 'country', 'disco',
            'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'
        ]
        
        # Audio features parameters for mel spectrogram
        self.sample_rate = 22050  # Original sample rate for GTZAN dataset
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        
        # Cache for mel spectrograms
        self.mel_cache = {}
        self.max_cache_size = 5
        
        # Genre characteristic features
        self.genre_features = {
            'blues': {'spectral_contrast_mean': 0.6, 'tempo_range': (70, 100), 'spectral_rolloff_mean': 0.5},
            'classical': {'spectral_contrast_mean': 0.3, 'tempo_range': (60, 120), 'spectral_rolloff_mean': 0.3},
            'country': {'spectral_contrast_mean': 0.5, 'tempo_range': (80, 130), 'spectral_rolloff_mean': 0.5},
            'disco': {'spectral_contrast_mean': 0.7, 'tempo_range': (110, 130), 'spectral_rolloff_mean': 0.7},
            'hiphop': {'spectral_contrast_mean': 0.7, 'tempo_range': (85, 115), 'spectral_rolloff_mean': 0.6},
            'jazz': {'spectral_contrast_mean': 0.4, 'tempo_range': (100, 170), 'spectral_rolloff_mean': 0.4},
            'metal': {'spectral_contrast_mean': 0.8, 'tempo_range': (100, 180), 'spectral_rolloff_mean': 0.8},
            'pop': {'spectral_contrast_mean': 0.6, 'tempo_range': (90, 130), 'spectral_rolloff_mean': 0.6},
            'reggae': {'spectral_contrast_mean': 0.5, 'tempo_range': (80, 110), 'spectral_rolloff_mean': 0.5},
            'rock': {'spectral_contrast_mean': 0.7, 'tempo_range': (100, 140), 'spectral_rolloff_mean': 0.7}
        }
        
        # Recent predictions for ensemble averaging
        self.recent_predictions = []
        self.max_recent_predictions = 3
        
        if model_path:
            self.load_model(model_path)
        else:
            print("Using CNN model with random weights for demonstration")
    
    def load_model(self, model_path):
        """Load pre-trained model weights"""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Loaded pre-trained model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Continue with random weights
            self.model.eval()
    
    def preprocess_audio(self, audio_data, sr=None):
        """
        Preprocess audio data for model input
        
        Args:
            audio_data: numpy array of audio samples
            sr: sample rate of audio data
            
        Returns:
            torch tensor ready for model input
        """
        start_time = time.time()
        
        # Create a unique hash for the audio data to use as cache key
        audio_hash = hashlib.md5(audio_data.tobytes()).hexdigest()
        
        # Check if we have this in cache
        if audio_hash in self.mel_cache:
            print(f"Using cached mel spectrogram (saved {time.time() - start_time:.2f}s)")
            return self.mel_cache[audio_hash], None
        
        # Original feature extraction
        if sr is not None and sr != self.sample_rate:
            # Resample if needed
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
        
        # Extract additional features for confidence boosting
        additional_features = self._extract_additional_features(audio_data)
            
        # Extract mel spectrogram features
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        # Normalize to [-1, 1] range
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
        
        # Add batch and channel dimensions
        mel_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0).unsqueeze(0)
        
        # Update cache, managing its size
        if len(self.mel_cache) >= self.max_cache_size:
            # Remove a random item
            self.mel_cache.pop(next(iter(self.mel_cache)))
        
        self.mel_cache[audio_hash] = mel_tensor
        
        print(f"Spectrogram processing took {time.time() - start_time:.2f}s")
        
        return mel_tensor, additional_features
    
    def _extract_additional_features(self, audio_data):
        """Extract additional features to help with confidence boosting"""
        features = {}
        
        # Extract tempo
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
        features['tempo'] = tempo
        
        # Extract spectral contrast (for timbre information)
        contrast = librosa.feature.spectral_contrast(
            y=audio_data, sr=self.sample_rate, n_bands=6
        )
        features['spectral_contrast_mean'] = np.mean(contrast)
        
        # Extract spectral rolloff (frequency below which is concentrated n% of spectral energy)
        rolloff = librosa.feature.spectral_rolloff(
            y=audio_data, sr=self.sample_rate
        )
        features['spectral_rolloff_mean'] = np.mean(rolloff) / self.sample_rate
        
        return features
    
    def _extract_audio_features(self, audio_data, sr=None):
        """Extract features from audio data for prediction"""
        
        # For efficiency, process at most 6 seconds of audio
        max_samples = 6 * self.sample_rate
        if len(audio_data) > max_samples:
            offset = (len(audio_data) - max_samples) // 2
            audio_data = audio_data[offset:offset + max_samples]
            
        return self.preprocess_audio(audio_data, sr)
    
    def _boost_confidence(self, probabilities, additional_features):
        """Boost confidence based on audio characteristics"""
        if additional_features is None:
            return probabilities
            
        boosted_probs = probabilities.clone()
        
        # Apply gentle boosting based on genre characteristics
        for i, genre in enumerate(self.genres):
            genre_chars = self.genre_features[genre]
            
            # Tempo match boost
            if genre_chars['tempo_range'][0] <= additional_features['tempo'] <= genre_chars['tempo_range'][1]:
                boosted_probs[i] *= 1.2  # 20% boost for matching tempo
            
            # Spectral contrast similarity boost
            contrast_diff = abs(genre_chars['spectral_contrast_mean'] - additional_features['spectral_contrast_mean'])
            if contrast_diff < 0.2:  # Close match
                boosted_probs[i] *= 1.1  # 10% boost
                
            # Spectral rolloff similarity boost
            rolloff_diff = abs(genre_chars['spectral_rolloff_mean'] - additional_features['spectral_rolloff_mean'])
            if rolloff_diff < 0.2:  # Close match
                boosted_probs[i] *= 1.1  # 10% boost
        
        # Renormalize
        return F.normalize(boosted_probs.unsqueeze(0), p=1, dim=1).squeeze(0)
    
    def _segment_predictions(self, audio_data, sr=None, num_segments=3):
        """Make predictions on multiple segments of audio"""
        if len(audio_data) < self.sample_rate * 3:  # If less than 3 seconds
            return self._single_prediction(audio_data, sr)
            
        segment_length = len(audio_data) // num_segments
        all_probabilities = []
        
        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            segment = audio_data[start:end]
            
            # Get prediction for this segment
            segment_probs = self._single_prediction(segment, sr)
            all_probabilities.append(segment_probs)
            
        # Average probabilities across segments
        avg_probs = torch.stack(all_probabilities).mean(dim=0)
        return avg_probs
    
    def _single_prediction(self, audio_data, sr=None):
        """Make a single prediction on the audio data"""
        # Extract features
        features, additional_features = self._extract_audio_features(audio_data, sr)
        
        # Make prediction
        with torch.no_grad():
            features = features.to(self.device)
            outputs = self.model(features)
            probabilities = F.softmax(outputs, dim=1)[0]
            
            # Apply confidence boosting
            if additional_features:
                probabilities = self._boost_confidence(probabilities, additional_features)
                
        return probabilities
    
    def _ensemble_with_recent(self, probabilities):
        """Ensemble with recent predictions for temporal smoothing"""
        # Add current prediction to recent list
        self.recent_predictions.append(probabilities)
        
        # Keep only the most recent predictions
        if len(self.recent_predictions) > self.max_recent_predictions:
            self.recent_predictions.pop(0)
            
        # Average all recent predictions
        if len(self.recent_predictions) > 1:
            ensemble_probs = torch.stack(self.recent_predictions).mean(dim=0)
            return ensemble_probs
        else:
            return probabilities
    
    def predict_genre(self, audio_data, sr=None):
        """
        Predict genre of audio data
        
        Args:
            audio_data: numpy array of audio samples
            sr: sample rate of audio data
            
        Returns:
            tuple of (genre, confidence)
        """
        # Ensure we have enough audio data to analyze (at least 1.5 sec)
        min_samples = int(1.5 * (sr or self.sample_rate))
        if len(audio_data) < min_samples:
            return "insufficient_audio", 0.0
            
        # Limit to max 30 seconds for efficiency
        max_samples = 30 * (sr or self.sample_rate)
        if len(audio_data) > max_samples:
            # Take the middle section
            start = (len(audio_data) - max_samples) // 2
            audio_data = audio_data[start:start + max_samples]
        
        # Use segment-based prediction for longer audio
        if len(audio_data) > self.sample_rate * 3:  # If more than 3 seconds
            probabilities = self._segment_predictions(audio_data, sr)
        else:
            # For shorter audio, use single prediction
            probabilities = self._single_prediction(audio_data, sr)
            
        # Ensemble with recent predictions for temporal smoothing
        ensemble_probs = self._ensemble_with_recent(probabilities)
            
        # Get predicted genre and confidence
        confidence, prediction = torch.max(ensemble_probs, 0)
        genre = self.genres[prediction.item()]
        confidence = confidence.item() * 100  # Convert to percentage
            
        return genre, confidence 