import numpy as np
import librosa
import time
import os
from typing import List, Dict, Tuple, Optional

try:
    import essentia.standard as es
    import essentia
    ESSENTIA_AVAILABLE = True
except ImportError:
    ESSENTIA_AVAILABLE = False
    print("Warning: Essentia not available. Falling back to basic detection.")

class InstrumentDetector:
    """
    Enhanced instrument detector using Essentia's pre-trained models
    with fallback to improved spectral analysis
    """
    
    def __init__(self):
        """
        Initialize the enhanced instrument detector
        """
        self.use_essentia = ESSENTIA_AVAILABLE
        
        if self.use_essentia:
            self._init_essentia_models()
        else:
            self._init_fallback_detector()
    
    def _init_essentia_models(self):
        """Initialize Essentia models and algorithms"""
        try:
            # Initialize Essentia algorithms
            self.windowing = es.Windowing(type='hann')
            self.spectrum = es.Spectrum()
            self.spectral_peaks = es.SpectralPeaks()
            self.spectral_centroid = es.SpectralCentroid()
            self.spectral_contrast = es.SpectralContrast()
            self.mfcc = es.MFCC()
            self.onset_detection = es.OnsetDetection(method='hfc')
            self.spectral_rolloff = es.SpectralRollOff()
            self.zerocrossingrate = es.ZeroCrossingRate()
            
            # Enhanced spectral features
            self.spectral_complexity = es.SpectralComplexity()
            self.dissonance = es.Dissonance()
            self.harmonicity = es.Harmonicity()
            
            print("✅ Essentia models initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Error initializing Essentia models: {e}")
            self.use_essentia = False
            self._init_fallback_detector()
    
    def _init_fallback_detector(self):
        """Initialize fallback detector with improved spectral analysis"""
        # Enhanced instrument profiles with better spectral characteristics
        self.instruments = {
            'piano': {
                'spectral_centroid_range': (800, 2500),
                'spectral_rolloff_range': (2000, 6000),
                'spectral_contrast_mean': 0.6,
                'harmonic_ratio_min': 0.7,
                'attack_time': 'short',
                'sustain_pattern': 'decay'
            },
            'violin': {
                'spectral_centroid_range': (1200, 3500),
                'spectral_rolloff_range': (3000, 8000),
                'spectral_contrast_mean': 0.75,
                'harmonic_ratio_min': 0.8,
                'attack_time': 'variable',
                'sustain_pattern': 'sustained'
            },
            'guitar': {
                'spectral_centroid_range': (600, 2000),
                'spectral_rolloff_range': (1500, 5000),
                'spectral_contrast_mean': 0.65,
                'harmonic_ratio_min': 0.6,
                'attack_time': 'medium',
                'sustain_pattern': 'decay'
            },
            'trumpet': {
                'spectral_centroid_range': (1000, 2800),
                'spectral_rolloff_range': (2500, 7000),
                'spectral_contrast_mean': 0.7,
                'harmonic_ratio_min': 0.75,
                'attack_time': 'short',
                'sustain_pattern': 'sustained'
            },
            'saxophone': {
                'spectral_centroid_range': (800, 2200),
                'spectral_rolloff_range': (2000, 6000),
                'spectral_contrast_mean': 0.68,
                'harmonic_ratio_min': 0.65,
                'attack_time': 'medium',
                'sustain_pattern': 'sustained'
            },
            'cello': {
                'spectral_centroid_range': (400, 1200),
                'spectral_rolloff_range': (800, 3000),
                'spectral_contrast_mean': 0.72,
                'harmonic_ratio_min': 0.78,
                'attack_time': 'variable',
                'sustain_pattern': 'sustained'
            },
            'drums': {
                'spectral_centroid_range': (2000, 8000),
                'spectral_rolloff_range': (4000, 15000),
                'spectral_contrast_mean': 0.9,
                'harmonic_ratio_min': 0.1,
                'attack_time': 'very_short',
                'sustain_pattern': 'percussive'
            },
            'bass': {
                'spectral_centroid_range': (200, 800),
                'spectral_rolloff_range': (400, 2000),
                'spectral_contrast_mean': 0.55,
                'harmonic_ratio_min': 0.7,
                'attack_time': 'medium',
                'sustain_pattern': 'sustained'
            }
        }
    
    def extract_essentia_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Extract features using Essentia algorithms"""
        features = {}
        
        try:
            # Ensure audio is in the right format
            audio_float = audio_data.astype(np.float32)
            
            # Frame-based analysis
            frame_size = 2048
            hop_size = 1024
            
            spectral_centroids = []
            spectral_contrasts = []
            spectral_rolloffs = []
            zero_crossing_rates = []
            mfccs = []
            complexities = []
            harmonicities = []
            
            for i in range(0, len(audio_float) - frame_size, hop_size):
                frame = audio_float[i:i + frame_size]
                
                # Apply windowing and get spectrum
                windowed_frame = self.windowing(frame)
                spectrum = self.spectrum(windowed_frame)
                
                # Extract spectral features
                centroid = self.spectral_centroid(spectrum)
                contrast = self.spectral_contrast(spectrum)
                rolloff = self.spectral_rolloff(spectrum)
                zcr = self.zerocrossingrate(frame)
                mfcc_coeffs = self.mfcc(spectrum)
                complexity = self.spectral_complexity(spectrum)
                
                # Get harmonicity
                peaks_frequencies, peaks_magnitudes = self.spectral_peaks(spectrum)
                if len(peaks_frequencies) > 0:
                    harmonicity = self.harmonicity(peaks_frequencies, peaks_magnitudes)
                else:
                    harmonicity = 0.0
                
                spectral_centroids.append(centroid)
                spectral_contrasts.append(np.mean(contrast))
                spectral_rolloffs.append(rolloff)
                zero_crossing_rates.append(zcr)
                mfccs.append(mfcc_coeffs)
                complexities.append(complexity)
                harmonicities.append(harmonicity)
            
            # Aggregate features
            features = {
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_contrast_mean': np.mean(spectral_contrasts),
                'spectral_contrast_std': np.std(spectral_contrasts),
                'spectral_rolloff_mean': np.mean(spectral_rolloffs),
                'spectral_rolloff_std': np.std(spectral_rolloffs),
                'zero_crossing_rate_mean': np.mean(zero_crossing_rates),
                'zero_crossing_rate_std': np.std(zero_crossing_rates),
                'mfcc_mean': np.mean(mfccs, axis=0),
                'mfcc_std': np.std(mfccs, axis=0),
                'spectral_complexity_mean': np.mean(complexities),
                'harmonicity_mean': np.mean(harmonicities),
                'harmonicity_std': np.std(harmonicities)
            }
            
        except Exception as e:
            print(f"Error extracting Essentia features: {e}")
            # Fallback to basic features
            return self.extract_fallback_features(audio_data, sr)
        
        return features
    
    def extract_fallback_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Extract features using librosa as fallback"""
        features = {}
        
        try:
            # Basic spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            
            # Enhanced features
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
            
            features = {
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'spectral_rolloff_std': np.std(spectral_rolloff),
                'spectral_contrast_mean': np.mean(spectral_contrast),
                'spectral_contrast_std': np.std(spectral_contrast),
                'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
                'zero_crossing_rate_std': np.std(zero_crossing_rate),
                'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                'spectral_bandwidth_std': np.std(spectral_bandwidth),
                'mfcc_mean': np.mean(mfccs, axis=1),
                'mfcc_std': np.std(mfccs, axis=1),
                'tempo': tempo,
                'chroma_mean': np.mean(chroma, axis=1),
                'chroma_std': np.std(chroma, axis=1)
            }
        except Exception as e:
            print(f"Error extracting fallback features: {e}")
            return {}
        
        return features
    
    def classify_instruments_essentia(self, features: Dict) -> List[Tuple[str, float]]:
        """Classify instruments using Essentia features with improved logic"""
        instrument_scores = {}
        
        # Enhanced classification logic
        centroid_mean = features.get('spectral_centroid_mean', 0)
        rolloff_mean = features.get('spectral_rolloff_mean', 0)
        contrast_mean = features.get('spectral_contrast_mean', 0)
        harmonicity = features.get('harmonicity_mean', 0)
        complexity = features.get('spectral_complexity_mean', 0)
        zcr_mean = features.get('zero_crossing_rate_mean', 0)
        
        # Piano detection - percussive attack, rich harmonics
        piano_score = 0
        if 800 <= centroid_mean <= 2500 and harmonicity > 0.6:
            piano_score += 0.6
        if contrast_mean > 0.5:
            piano_score += 0.3
        if complexity > 0.4:
            piano_score += 0.1
        instrument_scores['Piano'] = min(piano_score, 1.0)
        
        # Violin detection - high centroid, sustained harmonics
        violin_score = 0
        if 1200 <= centroid_mean <= 3500 and harmonicity > 0.7:
            violin_score += 0.7
        if contrast_mean > 0.6:
            violin_score += 0.2
        if zcr_mean < 0.1:  # Sustained notes
            violin_score += 0.1
        instrument_scores['Violin'] = min(violin_score, 1.0)
        
        # Guitar detection
        guitar_score = 0
        if 600 <= centroid_mean <= 2000 and harmonicity > 0.5:
            guitar_score += 0.6
        if 0.5 <= contrast_mean <= 0.8:
            guitar_score += 0.3
        if complexity > 0.3:
            guitar_score += 0.1
        instrument_scores['Guitar'] = min(guitar_score, 1.0)
        
        # Trumpet detection - bright, harmonic
        trumpet_score = 0
        if 1000 <= centroid_mean <= 2800 and harmonicity > 0.7:
            trumpet_score += 0.7
        if contrast_mean > 0.6:
            trumpet_score += 0.2
        if zcr_mean < 0.08:
            trumpet_score += 0.1
        instrument_scores['Trumpet'] = min(trumpet_score, 1.0)
        
        # Saxophone detection
        sax_score = 0
        if 800 <= centroid_mean <= 2200 and harmonicity > 0.6:
            sax_score += 0.6
        if 0.6 <= contrast_mean <= 0.8:
            sax_score += 0.3
        if complexity > 0.5:
            sax_score += 0.1
        instrument_scores['Saxophone'] = min(sax_score, 1.0)
        
        # Cello detection - lower frequencies, rich harmonics
        cello_score = 0
        if 400 <= centroid_mean <= 1200 and harmonicity > 0.7:
            cello_score += 0.7
        if contrast_mean > 0.6:
            cello_score += 0.2
        if zcr_mean < 0.06:
            cello_score += 0.1
        instrument_scores['Cello'] = min(cello_score, 1.0)
        
        # Drums detection - high centroid, low harmonicity
        drums_score = 0
        if centroid_mean > 2000 and harmonicity < 0.3:
            drums_score += 0.6
        if contrast_mean > 0.8:
            drums_score += 0.3
        if zcr_mean > 0.15:
            drums_score += 0.1
        instrument_scores['Drums'] = min(drums_score, 1.0)
        
        # Bass detection - very low frequencies
        bass_score = 0
        if centroid_mean < 800 and harmonicity > 0.6:
            bass_score += 0.7
        if rolloff_mean < 2000:
            bass_score += 0.2
        if zcr_mean < 0.05:
            bass_score += 0.1
        instrument_scores['Bass'] = min(bass_score, 1.0)
        
        # Sort by confidence and return
        sorted_instruments = sorted(instrument_scores.items(), key=lambda x: x[1], reverse=True)
        return [(name, score) for name, score in sorted_instruments if score > 0.3]
    
    def classify_instruments_fallback(self, features: Dict) -> List[Tuple[str, float]]:
        """Fallback classification using librosa features"""
        instrument_scores = {}
        
        centroid_mean = features.get('spectral_centroid_mean', 0)
        rolloff_mean = features.get('spectral_rolloff_mean', 0)
        contrast_mean = features.get('spectral_contrast_mean', 0)
        bandwidth_mean = features.get('spectral_bandwidth_mean', 0)
        zcr_mean = features.get('zero_crossing_rate_mean', 0)
        tempo = features.get('tempo', 120)
        
        # Simplified instrument detection
        for instrument, profile in self.instruments.items():
            score = 0
            
            # Check spectral centroid range
            centroid_range = profile['spectral_centroid_range']
            if centroid_range[0] <= centroid_mean <= centroid_range[1]:
                score += 0.4
            
            # Check spectral rolloff
            rolloff_range = profile['spectral_rolloff_range']
            if rolloff_range[0] <= rolloff_mean <= rolloff_range[1]:
                score += 0.3
            
            # Check spectral contrast
            if abs(contrast_mean - profile['spectral_contrast_mean']) < 0.2:
                score += 0.2
            
            # Special case for drums - high zero crossing rate
            if instrument == 'drums' and zcr_mean > 0.1:
                score += 0.1
            
            instrument_scores[instrument.title()] = min(score, 1.0)
        
        # Sort and filter
        sorted_instruments = sorted(instrument_scores.items(), key=lambda x: x[1], reverse=True)
        return [(name, score) for name, score in sorted_instruments if score > 0.35]
    
    def detect_instruments(self, audio_data: np.ndarray, sr: int, 
                          confidence_threshold: float = 0.4) -> List[str]:
        """
        Main instrument detection method
        
        Args:
            audio_data: Audio signal as numpy array
            sr: Sample rate
            confidence_threshold: Minimum confidence for instrument detection
            
        Returns:
            List of detected instrument names
        """
        start_time = time.time()
        
        try:
            # Extract features
            if self.use_essentia:
                features = self.extract_essentia_features(audio_data, sr)
                instrument_predictions = self.classify_instruments_essentia(features)
            else:
                features = self.extract_fallback_features(audio_data, sr)
                instrument_predictions = self.classify_instruments_fallback(features)
            
            # Filter by confidence threshold
            detected_instruments = [
                instrument for instrument, confidence in instrument_predictions 
                if confidence >= confidence_threshold
            ]
            
            # Limit to top 5 instruments
            detected_instruments = detected_instruments[:5]
            
            detection_time = time.time() - start_time
            
            print(f"Instrument detection completed in {detection_time:.3f} seconds")
            print(f"Detected instruments: {detected_instruments}")
            
            return detected_instruments
            
        except Exception as e:
            print(f"Error in instrument detection: {e}")
            return ['Unknown']
    
    def get_instrument_description(self, instrument: str) -> str:
        """Get description for an instrument"""
        descriptions = {
            'Piano': 'A keyboard instrument that produces sound by striking strings with hammers',
            'Violin': 'A bowed string instrument with a high pitch range and expressive capabilities',
            'Guitar': 'A plucked string instrument, fundamental to many genres of music',
            'Trumpet': 'A brass instrument with a bright, piercing sound and strong projection',
            'Saxophone': 'A woodwind instrument with a distinctive reedy sound and jazz associations',
            'Cello': 'A large bowed string instrument with a rich, warm tone in the lower register',
            'Drums': 'Percussion instruments that provide rhythmic foundation and dynamic accents',
            'Bass': 'A low-pitched string instrument that provides harmonic and rhythmic foundation'
        }
        return descriptions.get(instrument, f'{instrument} is a musical instrument') 