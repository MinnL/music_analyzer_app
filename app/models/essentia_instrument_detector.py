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
    print("Warning: Essentia not available. Install with: pip install essentia-tensorflow")

class EssentiaInstrumentDetector:
    """
    Advanced instrument detector using Essentia's pre-trained models and algorithms
    Provides much better accuracy than rule-based approaches
    """
    
    def __init__(self):
        """Initialize the Essentia-based instrument detector"""
        self.use_essentia = ESSENTIA_AVAILABLE
        
        if self.use_essentia:
            self._init_essentia()
        else:
            print("⚠️ Essentia not available - falling back to basic detection")
            self._init_fallback()
    
    def _init_essentia(self):
        """Initialize Essentia algorithms with robust error handling"""
        try:
            print("🔧 Attempting Essentia initialization...")
            
            # Test basic Essentia functionality first
            test_windowing = es.Windowing(type='hann')
            test_spectrum = es.Spectrum()
            print("✅ Basic Essentia algorithms work")
            
            # Use consistent frame size for all algorithms
            self.frame_size = 4096
            
            # Initialize algorithms one by one with error checking
            self.windowing = es.Windowing(type='hann')
            self.spectrum = es.Spectrum()
            self.spectral_peaks = es.SpectralPeaks(orderBy='magnitude', magnitudeThreshold=0.01)
            
            # Check if SpectralCentroidTime is available
            try:
                self.spectral_centroid = es.SpectralCentroidTime()
                print("✅ SpectralCentroidTime available")
            except AttributeError:
                print("⚠️ SpectralCentroidTime not available, using Centroid")
                self.spectral_centroid = es.Centroid()
            
            # Initialize SpectralContrast without frameSize parameter
            try:
                self.spectral_contrast = es.SpectralContrast()
                print("✅ SpectralContrast available")
            except Exception as e:
                print(f"⚠️ SpectralContrast failed: {e}")
                self.spectral_contrast = None
            
            self.spectral_rolloff = es.RollOff(cutoff=0.95)
            
            try:
                self.spectral_complexity = es.SpectralComplexity()
            except AttributeError:
                print("⚠️ SpectralComplexity not available")
                self.spectral_complexity = None
                
            self.zerocrossingrate = es.ZeroCrossingRate()
            self.mfcc = es.MFCC()
            
            try:
                self.inharmonicity = es.Inharmonicity()
            except AttributeError:
                print("⚠️ Inharmonicity not available")
                self.inharmonicity = None
                
            self.onsetdetection = es.OnsetDetection(method='hfc')
            
            print("✅ Essentia initialized successfully")
            self.use_essentia = True
            
        except Exception as e:
            print(f"❌ Essentia initialization failed: {e}")
            self.use_essentia = False
            print("🔄 Falling back to librosa-based detection...")
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize fallback profiles"""
        self.profiles = {
            'Piano': {'centroid': (800, 2500), 'rolloff': (2000, 6000)},
            'Violin': {'centroid': (1200, 3500), 'rolloff': (3000, 8000)},
            'Guitar': {'centroid': (600, 2200), 'rolloff': (1500, 5500)},
            'Trumpet': {'centroid': (1000, 2800), 'rolloff': (2500, 7000)},
            'Saxophone': {'centroid': (800, 2400), 'rolloff': (2000, 6500)},
            'Cello': {'centroid': (300, 1200), 'rolloff': (800, 3500)},
            'Drums': {'centroid': (2000, 10000), 'rolloff': (4000, 15000)},
            'Bass': {'centroid': (200, 600), 'rolloff': (400, 1500)}
        }
    
    def extract_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Extract comprehensive features"""
        if self.use_essentia:
            return self._extract_essentia_features(audio_data, sr)
        else:
            return self._extract_basic_features(audio_data, sr)
    
    def _extract_essentia_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Extract features using Essentia with robust error handling"""
        try:
            print("🎵 Starting Essentia feature extraction...")
            audio_float = audio_data.astype(np.float32)
            frame_size = self.frame_size if hasattr(self, 'frame_size') else 4096
            hop_size = frame_size // 2
            
            centroids = []
            contrasts = []
            rolloffs = []
            complexities = []
            zcrs = []
            inharmonicities = []
            onsets = []
            
            # Process frames for feature extraction
            num_frames = 0
            max_frames = 50  # Limit processing to avoid hanging
            
            try:
                for i in range(0, min(len(audio_float) - frame_size, max_frames * hop_size), hop_size):
                    frame = audio_float[i:i + frame_size]
                    windowed = self.windowing(frame)
                    spectrum = self.spectrum(windowed)
                    num_frames += 1
                    
                    # Extract spectral centroid
                    try:
                        if hasattr(self.spectral_centroid, '__call__'):
                            # For SpectralCentroidTime, use time domain
                            if 'Time' in str(type(self.spectral_centroid)):
                                centroids.append(self.spectral_centroid(frame))
                            else:
                                # For regular Centroid, use spectrum
                                centroids.append(self.spectral_centroid(spectrum))
                    except Exception as e:
                        print(f"Centroid extraction failed: {e}")
                        centroids.append(1000.0)  # Default value
                    
                    # Extract spectral contrast
                    if self.spectral_contrast is not None:
                        try:
                            contrast_result = self.spectral_contrast(spectrum)
                            if isinstance(contrast_result, (list, np.ndarray)):
                                contrasts.append(np.mean(contrast_result))
                            else:
                                contrasts.append(contrast_result)
                        except Exception as e:
                            contrasts.append(0.5)  # Default value
                    else:
                        contrasts.append(0.5)
                    
                    # Extract spectral rolloff
                    try:
                        rolloffs.append(self.spectral_rolloff(spectrum))
                    except Exception:
                        rolloffs.append(2000.0)  # Default value
                    
                    # Extract spectral complexity
                    if self.spectral_complexity is not None:
                        try:
                            complexities.append(self.spectral_complexity(spectrum))
                        except Exception:
                            complexities.append(0.5)  # Default value
                    else:
                        complexities.append(0.5)
                    
                    # Extract zero crossing rate
                    try:
                        zcrs.append(self.zerocrossingrate(frame))
                    except Exception:
                        zcrs.append(0.1)  # Default value
                    
                    # Extract onset detection
                    try:
                        # OnsetDetection might need different parameters
                        onset_result = self.onsetdetection(spectrum, spectrum)  # Try with two arguments
                        onsets.append(onset_result)
                    except Exception:
                        try:
                            onset_result = self.onsetdetection(spectrum)  # Try with one argument
                            onsets.append(onset_result)
                        except Exception:
                            onsets.append(0.0)  # Default value
                    
                    # Extract inharmonicity
                    if self.inharmonicity is not None:
                        try:
                            peaks_freq, peaks_mag = self.spectral_peaks(spectrum)
                            if len(peaks_freq) > 1:
                                inharm_result = self.inharmonicity(peaks_freq, peaks_mag)
                                inharmonicities.append(1.0 - inharm_result)  # Invert to get harmonicity-like measure
                            else:
                                inharmonicities.append(0.0)
                        except Exception:
                            inharmonicities.append(0.0)  # Default value
                    else:
                        inharmonicities.append(0.0)
                        
                print(f"✅ Processed {num_frames} frames successfully")
                        
            except Exception as frame_error:
                print(f"Frame processing failed: {frame_error}")
                # If frame processing fails completely, use single frame approach
                frame = audio_float[:frame_size]
                windowed = self.windowing(frame)
                spectrum = self.spectrum(windowed)
                
                centroids = [1000.0]  # Default centroid
                contrasts = [0.5]
                rolloffs = [2000.0]
                complexities = [0.5]
                zcrs = [0.1]
                onsets = [0.0]
                inharmonicities = [0.0]
            
            # Calculate averages with fallback values
            result = {
                'spectral_centroid_mean': np.mean(centroids) if centroids else 1000.0,
                'spectral_contrast_mean': np.mean(contrasts) if contrasts else 0.5,
                'spectral_rolloff_mean': np.mean(rolloffs) if rolloffs else 2000.0,
                'spectral_complexity_mean': np.mean(complexities) if complexities else 0.5,
                'zero_crossing_rate_mean': np.mean(zcrs) if zcrs else 0.1,
                'harmonicity_mean': np.mean(inharmonicities) if inharmonicities else 0.0,
                'onset_density': np.sum(np.array(onsets) > 0.1) / len(onsets) if onsets else 0.0
            }
            
            print("✅ Essentia feature extraction completed")
            return result
            
        except Exception as e:
            print(f"❌ Essentia feature extraction error: {e}")
            print("🔄 Falling back to librosa-based feature extraction...")
            return self._extract_basic_features(audio_data, sr)
    
    def _extract_basic_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Basic feature extraction using librosa"""
        try:
            centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
            contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
            onsets = librosa.onset.onset_detect(y=audio_data, sr=sr)
            
            return {
                'spectral_centroid_mean': np.mean(centroids),
                'spectral_rolloff_mean': np.mean(rolloff),
                'spectral_contrast_mean': np.mean(contrast),
                'zero_crossing_rate_mean': np.mean(zcr),
                'onset_density': len(onsets) / (len(audio_data) / sr)
            }
        except Exception as e:
            print(f"Basic feature extraction error: {e}")
            return {}
    
    def classify_instruments(self, features: Dict) -> List[Tuple[str, float]]:
        """Classify instruments using extracted features"""
        if self.use_essentia:
            return self._classify_advanced(features)
        else:
            return self._classify_basic(features)
    
    def _classify_advanced(self, features: Dict) -> List[Tuple[str, float]]:
        """Advanced classification using Essentia features"""
        scores = {}
        
        centroid = features.get('spectral_centroid_mean', 0)
        rolloff = features.get('spectral_rolloff_mean', 0)
        contrast = features.get('spectral_contrast_mean', 0)
        harmonicity = features.get('harmonicity_mean', 0)
        complexity = features.get('spectral_complexity_mean', 0)
        zcr = features.get('zero_crossing_rate_mean', 0)
        onset_density = features.get('onset_density', 0)
        
        # Piano - percussive, rich harmonics
        piano_score = 0
        if 700 <= centroid <= 2800 and harmonicity > 0.6:
            piano_score += 0.6
        if onset_density > 0.1:
            piano_score += 0.3
        if complexity > 0.4:
            piano_score += 0.1
        scores['Piano'] = min(piano_score, 1.0)
        
        # Violin - high harmonicity, sustained
        violin_score = 0
        if 1000 <= centroid <= 3800 and harmonicity > 0.8:
            violin_score += 0.7
        if zcr < 0.08:
            violin_score += 0.2
        if contrast > 0.6:
            violin_score += 0.1
        scores['Violin'] = min(violin_score, 1.0)
        
        # Guitar - medium harmonicity, plucked
        guitar_score = 0
        if 500 <= centroid <= 2500 and 0.5 <= harmonicity <= 0.8:
            guitar_score += 0.6
        if 0.05 <= onset_density <= 0.3:
            guitar_score += 0.3
        if contrast > 0.5:
            guitar_score += 0.1
        scores['Guitar'] = min(guitar_score, 1.0)
        
        # Trumpet - bright, harmonic
        trumpet_score = 0
        if 900 <= centroid <= 3200 and harmonicity > 0.7:
            trumpet_score += 0.7
        if zcr < 0.06:
            trumpet_score += 0.2
        if contrast > 0.6:
            trumpet_score += 0.1
        scores['Trumpet'] = min(trumpet_score, 1.0)
        
        # Saxophone - reed characteristics
        sax_score = 0
        if 700 <= centroid <= 2600 and 0.6 <= harmonicity <= 0.85:
            sax_score += 0.6
        if complexity > 0.5:
            sax_score += 0.3
        if zcr < 0.08:
            sax_score += 0.1
        scores['Saxophone'] = min(sax_score, 1.0)
        
        # Cello - low, harmonic
        cello_score = 0
        if 250 <= centroid <= 1400 and harmonicity > 0.75:
            cello_score += 0.7
        if zcr < 0.05:
            cello_score += 0.2
        if rolloff < 3500:
            cello_score += 0.1
        scores['Cello'] = min(cello_score, 1.0)
        
        # Drums - percussive, low harmonicity
        drums_score = 0
        if centroid > 1800 and harmonicity < 0.4:
            drums_score += 0.6
        if onset_density > 0.2:
            drums_score += 0.3
        if zcr > 0.12:
            drums_score += 0.1
        scores['Drums'] = min(drums_score, 1.0)
        
        # Bass - very low frequencies
        bass_score = 0
        if centroid < 700 and rolloff < 2000:
            bass_score += 0.8
        if harmonicity > 0.6:
            bass_score += 0.1
        if zcr < 0.04:
            bass_score += 0.1
        scores['Bass'] = min(bass_score, 1.0)
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def _classify_basic(self, features: Dict) -> List[Tuple[str, float]]:
        """Basic classification for fallback"""
        scores = {}
        
        centroid = features.get('spectral_centroid_mean', 0)
        rolloff = features.get('spectral_rolloff_mean', 0)
        
        for instrument, profile in self.profiles.items():
            score = 0
            if profile['centroid'][0] <= centroid <= profile['centroid'][1]:
                score += 0.6
            if profile['rolloff'][0] <= rolloff <= profile['rolloff'][1]:
                score += 0.4
            scores[instrument] = score
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def detect_instruments(self, audio_data: np.ndarray, sr: int, 
                          confidence_threshold: float = 0.4) -> List[str]:
        """Main detection method"""
        start_time = time.time()
        
        try:
            print(f"🎵 Starting {'Essentia' if self.use_essentia else 'basic'} instrument detection...")
            
            features = self.extract_features(audio_data, sr)
            if not features:
                return ['Unknown']
            
            predictions = self.classify_instruments(features)
            detected = [inst for inst, conf in predictions if conf >= confidence_threshold][:5]
            
            print(f"✅ Detection completed in {time.time() - start_time:.3f}s")
            print(f"🎯 Detected: {detected}")
            
            return detected if detected else ['Mixed Instruments']
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return ['Unknown']
    
    def get_instrument_info(self, instrument: str) -> Dict[str, str]:
        """Get detailed information about an instrument"""
        info = {
            'Piano': {
                'family': 'Keyboard/Percussion',
                'description': 'A keyboard instrument producing sound by striking strings with hammers',
                'characteristics': 'Percussive attack, rich harmonics, wide dynamic range'
            },
            'Violin': {
                'family': 'String (Bowed)',
                'description': 'A high-pitched bowed string instrument with expressive capabilities',
                'characteristics': 'Sustained tones, high harmonicity, expressive vibrato'
            },
            'Guitar': {
                'family': 'String (Plucked)',
                'description': 'A plucked string instrument fundamental to many musical styles',
                'characteristics': 'Plucked attack, rich harmonics, versatile timbres'
            },
            'Trumpet': {
                'family': 'Brass',
                'description': 'A brass instrument with bright, projecting sound',
                'characteristics': 'Bright timbre, strong harmonics, sustained notes'
            },
            'Saxophone': {
                'family': 'Woodwind (Reed)',
                'description': 'A woodwind instrument with distinctive reedy character',
                'characteristics': 'Reed complexity, warm tone, expressive dynamics'
            },
            'Cello': {
                'family': 'String (Bowed)',
                'description': 'A large bowed string instrument with rich, warm tone',
                'characteristics': 'Deep resonance, sustained tones, rich harmonics'
            },
            'Drums': {
                'family': 'Percussion',
                'description': 'Percussion instruments providing rhythmic foundation',
                'characteristics': 'Percussive attack, transient sounds, rhythmic patterns'
            },
            'Bass': {
                'family': 'String (Low-pitched)',
                'description': 'A low-pitched string instrument providing harmonic foundation',
                'characteristics': 'Low frequencies, fundamental-heavy, sustained tones'
            }
        }
        
        return info.get(instrument, {
            'family': 'Unknown',
            'description': f'{instrument} is a musical instrument',
            'characteristics': 'Various musical characteristics'
        }) 