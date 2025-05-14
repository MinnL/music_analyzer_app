import numpy as np
import librosa
import torch
import torchaudio
from sklearn.preprocessing import StandardScaler
import os
import json
from .models.genre_classifier import GenreClassifier
from .models.advanced_genre_classifier import AdvancedGenreClassifier
import time
import hashlib
from .models.high_confidence_classifier import HighConfidenceClassifier
from dash import html
from .models.instrument_detector import InstrumentDetector

class MusicAnalyzer:
    """
    Class for analyzing audio to identify genre and components
    """
    
    # Set default to use the original GTZAN-based classifier
    def __init__(self, sample_rate=22050, use_gtzan_model=True):
        """
        Initialize the music analyzer
        
        Args:
            sample_rate: Sample rate for audio analysis
            use_gtzan_model: Whether to use the original Genre Classifier with GTZAN weights.
        """
        # Initialize attributes
        self.sample_rate = sample_rate
        self.use_gtzan_model = use_gtzan_model # Store the preference
        # Remove the old flag if it exists (optional, for clarity)
        # if hasattr(self, 'use_advanced_model'):
        #    del self.use_advanced_model
        
        # Define genres directly in the class
        self.genres = [
            'blues', 'classical', 'country', 'disco',
            'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'
        ]
        
        # Initialize genre classifier to None
        self.genre_classifier = None
        
        # Initialize instrument detector
        self.instrument_detector = InstrumentDetector()
        
        # Initialize component descriptions (for UI details)
        self.component_descriptions = {
            "rhythm": {
                "tempo": "The speed or pace of the music, measured in beats per minute (BPM).",
                "beat_strength": "How pronounced or emphasized the beats are in the music.",
                "rhythm_complexity": "How complex or varied the rhythm patterns are.",
                "time_signature": "The organization of beats into measures (e.g., 4/4, 3/4).",
                "syncopation": "Placement of rhythmic stresses or accents where they wouldn't normally occur.",
                "groove": "The sense of movement and pattern that makes music feel good.",
                "classical": "Often has clear, structured rhythms with moderate tempos and minimal syncopation.",
                "electronic": "Features precise, machine-like rhythms with strong beats and often complex patterns.",
                "jazz": "Characterized by swing rhythms, syncopation, and often complex time signatures.",
                "rock": "Typically has strong, steady beats with emphasis on 2nd and 4th beats of 4/4 time.",
                "hip-hop": "Features strong beats with emphasis on bass and often complex syncopated patterns."
            },
            "melody": {
                "pitch_range": "The range between the highest and lowest notes in the melody.",
                "melodic_complexity": "How complex or varied the melody patterns are.",
                "key_signature": "The set of sharps or flats in the music, indicating the key.",
                "modality": "Whether the melody is in a major or minor key, affecting its emotional quality.",
                "melodic_contour": "The shape or direction of the melody as it rises and falls.",
                "phrasing": "How the melody is divided into musical sentences or phrases.",
                "classical": "Often features complex, developed melodies with wide pitch ranges and formal structures.",
                "jazz": "Characterized by improvisation, blue notes, and complex melodic structures.",
                "pop": "Typically features catchy, repetitive melodies with simple structures designed for memorability.",
                "country": "Often features storytelling melodies with twangy vocal styles and simple structures.",
                "blues": "Characterized by blue notes, bends, and specific scale patterns that create its distinctive sound."
            },
            "instrumentation": {
                "timbre": "The tonal quality or 'color' of instruments or voices.",
                "instrument_variety": "The range of different instruments used in the music.",
                "sonic_texture": "The layering and interaction of different sounds in the music.",
                "dominant_instruments": "The instruments that are most prominent in the music.",
                "acoustic_vs_electric": "Whether the music primarily uses acoustic or electric/electronic instruments.",
                "classical": "Primarily orchestral instruments like strings, woodwinds, brass, and percussion.",
                "rock": "Typically features electric guitars, bass guitar, drums, and vocals.",
                "electronic": "Uses synthesizers, drum machines, samplers, and digital processing.",
                "jazz": "Often includes saxophone, trumpet, piano, double bass, and drums.",
                "hip-hop": "Features drum machines, samplers, turntables, and often digital production.",
                "metal": "Heavy distorted guitars, aggressive drums, bass, and often harsh vocals."
            }
        }
        
        # Genre descriptions and characteristics
        self.genre_descriptions = {
            'blues': {
                'description': "Blues is characterized by its distinctive 'blue' notes, call-and-response patterns, and emotional expression.",
                'rhythm': "Typically steady, moderate tempo with shuffle or straight patterns, often using 12-bar form.",
                'melody': "Features flattened 'blue' notes, vocal-like bending, and expressive microtonal variations.",
                'instrumentation': "Often includes guitar, harmonica, piano, bass, and drums with emphasis on guitar solos and vocals."
            },
            'classical': {
                'description': "Classical music encompasses a broad period of Western art music with formal structures and orchestral arrangements.",
                'rhythm': "Precise rhythmic patterns that vary from slow, measured tempos to complex, faster movements.",
                'melody': "Complex, fully developed melodic themes with formal structure and harmonic richness.",
                'instrumentation': "Full orchestra with strings, woodwinds, brass, and percussion sections, often with featured solo instruments."
            },
            'country': {
                'description': "Country music originates from American folk and western traditions with storytelling lyrics and distinctive vocal styles.",
                'rhythm': "Steady, moderate tempos with clear beat patterns, often incorporating elements of folk and dance rhythms.",
                'melody': "Simple, memorable melodic lines emphasizing storytelling, often with distinctive twang in vocal delivery.",
                'instrumentation': "Traditional acoustic instruments like guitar, fiddle, banjo, steel guitar, with modern country adding electric elements."
            },
            'disco': {
                'description': "Disco emerged in the 1970s as dance music characterized by steady beats, orchestral elements, and infectious energy.",
                'rhythm': "Driving four-on-the-floor beat (bass drum on every beat) at 110-130 BPM, with prominent hi-hats.",
                'melody': "Catchy vocal hooks and melodic phrases designed for danceability and mass appeal.",
                'instrumentation': "Blend of orchestra, electric bass with emphatic lines, synthesizers, and rhythm guitar with characteristic wah-wah effects."
            },
            'hiphop': {
                'description': "Hip-hop centers around rhythmic vocals delivered over beats, with origins in urban African American communities.",
                'rhythm': "Strong beats with emphasis on bass, often using sampled or programmed drum patterns with syncopation.",
                'melody': "Often uses samples from other songs, with melodic elements supporting rhythmic vocal delivery.",
                'instrumentation': "Drum machines, samplers, turntables, synthesizers, with modern production incorporating various electronic elements."
            },
            'jazz': {
                'description': "Jazz is known for improvisation, syncopation, swing feel, and complex harmonies developed from African American musical traditions.",
                'rhythm': "Swung rhythms with emphasis on off-beats, or complex polyrhythms in more advanced forms.",
                'melody': "Improvisation over chord progressions, with blue notes, modal explorations, and complex phrases.",
                'instrumentation': "Typically includes saxophone, trumpet, piano, double bass, and drums, with various combinations in different jazz subgenres."
            },
            'metal': {
                'description': "Metal is characterized by heavily distorted guitars, emphatic beats, dense sound, and often aggressive or virtuosic performance.",
                'rhythm': "Driving rhythms, often with double bass drums, complex time signatures in progressive forms.",
                'melody': "Power chords, guitar solos, vocal styles ranging from clean to growled or screamed.",
                'instrumentation': "Heavy distorted guitars, aggressive drums, bass, and vocals, often with technical, virtuosic playing."
            },
            'pop': {
                'description': "Pop music aims for mass appeal with catchy hooks, simple structures, and accessible sounds that reflect contemporary trends.",
                'rhythm': "Clear, steady beats emphasizing danceability, typically between 90-130 BPM.",
                'melody': "Memorable, repetitive hooks and chorus sections designed for easy listening and sing-along appeal.",
                'instrumentation': "Varies with trends, combining electronic elements, programmed drums, and conventional instruments with polished production."
            },
            'reggae': {
                'description': "Reggae originated in Jamaica with distinctive offbeat rhythms, bass-heavy sound, and often socially conscious lyrics.",
                'rhythm': "Emphasis on the offbeat ('skank'), with bass playing a central role in defining the rhythm.",
                'melody': "Often simple, memorable melodic lines delivered with distinctive Jamaican vocal styles.",
                'instrumentation': "Emphasizes bass guitar and drums ('riddim'), with rhythm guitar playing offbeat chords, organs, and horns."
            },
            'rock': {
                'description': "Rock music is centered around electric guitar, strong beats, and often rebellious themes, evolved from rock and roll.",
                'rhythm': "Strong backbeat (emphasis on beats 2 and 4), typically in 4/4 time with variations in tempo and intensity.",
                'melody': "Guitar-driven riffs, vocal melodies ranging from simple to complex, often with verse-chorus structure.",
                'instrumentation': "Electric guitars, bass guitar, drums, and vocals, with various additions depending on subgenre."
            }
        }
        
        # Create a cache to avoid reprocessing the same audio
        self.analysis_cache = {}
        self.cache_size_limit = 5  # Only keep last 5 analyses
        
        # Load or initialize model
        self._load_or_init_model()
        
        # Previous analysis results
        self.previous_analysis = None
        
    def _load_or_init_model(self):
        """Load or initialize the appropriate genre classification model based on instance flags."""
        if self.genre_classifier is not None:
            return # Already loaded

        try:
            if hasattr(self, 'use_gtzan_model') and self.use_gtzan_model:
                # Load the original GenreClassifier with the GTZAN weights
                print("Using Original Genre Classifier with GTZAN pre-trained weights.")
                model_path = "app/models/pretrained/gtzan_model.pt"
                if os.path.exists(model_path):
                    self.genre_classifier = GenreClassifier(model_path=model_path)
                    print(f"Successfully loaded model from {model_path}")
                else:
                    print(f"ERROR: Pre-trained model file not found at {model_path}. Initializing with random weights.")
                    self.genre_classifier = GenreClassifier() # Fallback to random weights

            elif hasattr(self, 'use_advanced_model') and self.use_advanced_model:
                 # Load the Advanced Genre Classifier (currently uses random weights)
                 print("Using Advanced Genre Classifier (VGGish-based - currently with random weights).")
                 self.genre_classifier = AdvancedGenreClassifier(num_genres=len(self.genres))
                 # TODO: Implement loading of actual pre-trained weights for Advanced model if available

            elif hasattr(self, 'use_high_confidence') and self.use_high_confidence:
                 # Load the High Confidence Classifier (feature-based)
                 print("Using High Confidence Genre Classifier (feature-based).")
                 self.genre_classifier = HighConfidenceClassifier()
            
            else:
                # Default fallback if no specific flag is set (or logic error)
                print("Warning: No specific classifier preference set. Falling back to GTZAN model check.")
                model_path = "app/models/pretrained/gtzan_model.pt"
                if os.path.exists(model_path):
                    self.genre_classifier = GenreClassifier(model_path=model_path)
                else:
                     self.genre_classifier = GenreClassifier() # Random weights if default fails

        except Exception as e:
            print(f"CRITICAL Error initializing genre classifier: {e}")
            self.genre_classifier = None # Indicate failure
        
    def _init_dummy_model(self):
        """Initialize a dummy model for demonstration purposes"""
        # This is kept for backward compatibility but will not be used
        print("Using dummy model for demonstration purposes")
        
    def extract_features(self, audio_data):
        """
        Extract audio features from raw audio data
        
        Args:
            audio_data: Numpy array of audio samples
            
        Returns:
            Dictionary of extracted features
        """
        # If audio data is empty, return empty features
        if len(audio_data) == 0:
            return {
                "tempo": 0.0,
                "spectral_centroid": np.array([]),
                "spectral_rolloff": np.array([]),
                "spectral_contrast": np.array([]),
                "chroma": np.array([]),
                "mfcc": np.array([])
            }
            
        # Check if audio is in cache using hash
        audio_hash = hash(audio_data[:min(len(audio_data), self.sample_rate * 3)].tobytes())
        if audio_hash in self.analysis_cache and 'features' in self.analysis_cache[audio_hash]:
            # print("Using cached features")
            return self.analysis_cache[audio_hash]['features']
        
        # Compute basic audio features using librosa
        try:
            # Use shorter audio segment (max 6 seconds)
            max_length = min(len(audio_data), self.sample_rate * 6)
            audio_segment = audio_data[:max_length]
            
            # Tempo and beat information - use smaller frames
            onset_env = librosa.onset.onset_strength(
                y=audio_segment, 
                sr=self.sample_rate,
                hop_length=512  # Smaller hop length
            )
            tempo, _ = librosa.beat.beat_track(
                onset_envelope=onset_env, 
                sr=self.sample_rate,
                hop_length=512
            )
            
            # Ensure tempo is a scalar
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo.item() if tempo.size == 1 else tempo.mean())
            
            # Spectral features with reduced frame size
            hop_length = 1024  # Larger hop length for faster processing
            
            # Use a smaller fft size for faster computation
            n_fft = 1024
            
            # Only compute essential spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_segment, 
                sr=self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length
            )[0]
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_segment, 
                sr=self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length
            )[0]
            
            # Calculate spectral contrast with reduced complexity
            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio_segment, 
                sr=self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_bands=4  # Fewer bands for faster processing
            )
            
            # Tonal features with reduced complexity
            chroma = librosa.feature.chroma_stft(
                y=audio_segment, 
                sr=self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_chroma=12
            )
            
            # Fewer MFCCs for faster processing
            mfcc = librosa.feature.mfcc(
                y=audio_segment, 
                sr=self.sample_rate, 
                n_mfcc=10,  # Fewer coefficients
                n_fft=n_fft,
                hop_length=hop_length
            )
            
            features = {
                "tempo": tempo,
                "spectral_centroid": spectral_centroid,
                "spectral_rolloff": spectral_rolloff,
                "spectral_contrast": spectral_contrast,
                "chroma": chroma,
                "mfcc": mfcc
            }
            
            # Cache the features
            if audio_hash not in self.analysis_cache:
                self.analysis_cache[audio_hash] = {}
            self.analysis_cache[audio_hash]['features'] = features
            
            # Remove oldest entries if cache gets too large
            if len(self.analysis_cache) > self.cache_size_limit:
                oldest_key = next(iter(self.analysis_cache))
                del self.analysis_cache[oldest_key]
                
            return features
        except Exception as e:
            print(f"Error extracting features: {e}")
            return {
                "tempo": 0.0,
                "spectral_centroid": np.array([]),
                "spectral_rolloff": np.array([]),
                "spectral_contrast": np.array([]),
                "chroma": np.array([]),
                "mfcc": np.array([])
            }
            
    def analyze_components(self, features):
        """
        Analyze audio components (rhythm, melody, instrumentation)
        
        Args:
            features: Dictionary of audio features
            
        Returns:
            Dictionary of component analyses
        """
        # Simplified analysis for better performance
        
        # Analyze rhythm
        rhythm = {}
        if features["tempo"] > 0:
            # Ensure tempo is a scalar value
            tempo = features["tempo"]
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo.item() if tempo.size == 1 else tempo.mean())
                
            rhythm["tempo"] = tempo
            
            # Simple categorization
            if tempo < 70:
                rhythm["tempo_category"] = "Slow"
            elif tempo < 120:
                rhythm["tempo_category"] = "Medium"
            else:
                rhythm["tempo_category"] = "Fast"
                
            # Simplified rhythm complexity estimation
            if len(features["spectral_contrast"]) > 0:
                # Use mean instead of std for faster calculation
                rhythm_complexity = np.mean(np.mean(features["spectral_contrast"], axis=1))
                rhythm["complexity"] = float(rhythm_complexity)
                
                if rhythm_complexity < 0.4:
                    rhythm["complexity_category"] = "Simple"
                elif rhythm_complexity < 0.8:
                    rhythm["complexity_category"] = "Moderate"
                else:
                    rhythm["complexity_category"] = "Complex"
        else:
            rhythm = {"tempo": 0.0, "tempo_category": "Unknown", "complexity": 0.0, "complexity_category": "Unknown"}
            
        # Analyze melody - simplified
        melody = {}
        if features["chroma"].size > 0:
            # Get just top note instead of all 3
            chroma_mean = np.mean(features["chroma"], axis=1)
            dominant_note_idx = np.argmax(chroma_mean)
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            melody["dominant_notes"] = [note_names[dominant_note_idx % 12]]
            
            # Simplified pitch variety
            pitch_variety = float(np.max(chroma_mean) - np.min(chroma_mean))
            melody["pitch_variety"] = pitch_variety
            
            if pitch_variety < 0.1:
                melody["variety_category"] = "Low"
            elif pitch_variety < 0.2:
                melody["variety_category"] = "Medium"
            else:
                melody["variety_category"] = "High"
                
            # Simplified modality detection
            major_third_idx = (dominant_note_idx + 4) % 12
            minor_third_idx = (dominant_note_idx + 3) % 12
            
            if chroma_mean[major_third_idx] > chroma_mean[minor_third_idx]:
                melody["modality"] = "Major"
            else:
                melody["modality"] = "Minor"
        else:
            melody = {
                "dominant_notes": [], 
                "pitch_variety": 0.0, 
                "variety_category": "Unknown",
                "modality": "Unknown"
            }
            
        # Analyze instrumentation - simplified
        instrumentation = {}
        if features["mfcc"].size > 0:
            # Simplified brightness calculation
            if features["spectral_centroid"].size > 0:
                brightness = np.mean(features["spectral_centroid"]) / (self.sample_rate/2)
                instrumentation["brightness"] = float(brightness)
                
                if brightness < 0.3:
                    instrumentation["brightness_category"] = "Dark/Warm"
                elif brightness < 0.6:
                    instrumentation["brightness_category"] = "Balanced"
                else:
                    instrumentation["brightness_category"] = "Bright/Sharp"
            else:
                instrumentation["brightness"] = 0.0
                instrumentation["brightness_category"] = "Unknown"
                
            # Simplified contrast calculation
            if features["spectral_contrast"].size > 0:
                # Use mean instead of complex calculation
                contrast = float(np.mean(features["spectral_contrast"]))
                instrumentation["contrast"] = contrast
                
                if contrast < 20:
                    instrumentation["contrast_category"] = "Blended/Smooth"
                elif contrast < 40:
                    instrumentation["contrast_category"] = "Balanced"
                else:
                    instrumentation["contrast_category"] = "Distinct/Clear"
            else:
                instrumentation["contrast"] = 0.0
                instrumentation["contrast_category"] = "Unknown"
                
            # Simplified timbre complexity (faster calculation)
            timbre_complexity = float(np.mean(features["mfcc"]))
            instrumentation["complexity"] = timbre_complexity
            
            if timbre_complexity < -5:
                instrumentation["complexity_category"] = "Simple/Pure"
            elif timbre_complexity < 5:
                instrumentation["complexity_category"] = "Moderate"
            else:
                instrumentation["complexity_category"] = "Complex/Rich"
        else:
            instrumentation = {
                "brightness": 0.0, 
                "brightness_category": "Unknown",
                "contrast": 0.0, 
                "contrast_category": "Unknown",
                "complexity": 0.0, 
                "complexity_category": "Unknown"
            }
            
        return {
            "rhythm": rhythm,
            "melody": melody,
            "instrumentation": instrumentation
        }
    
    def classify_genre(self, audio_data):
        """
        Classify music genre based on audio data
        
        Args:
            audio_data: Raw audio data as numpy array
            
        Returns:
            Tuple of (genre, confidence)
        """
        # If audio_data is empty, return unknown
        if len(audio_data) == 0:
            return "Unknown", 0.0
            
        # Check if result is in cache
        audio_hash = hash(audio_data[:min(len(audio_data), self.sample_rate * 3)].tobytes())
        if audio_hash in self.analysis_cache and 'genre' in self.analysis_cache[audio_hash]:
            # print("Using cached genre")
            return self.analysis_cache[audio_hash]['genre'], self.analysis_cache[audio_hash]['confidence']
            
        # Use the pretrained model to predict genre
        try:
            # Ensure model is loaded
            if self.genre_classifier is None:
                self._load_or_init_model()
                
            genre, confidence = self.genre_classifier.predict_genre(
                audio_data=audio_data,
                sr=self.sample_rate
            )
            
            # Cache the result
            if audio_hash not in self.analysis_cache:
                self.analysis_cache[audio_hash] = {}
            self.analysis_cache[audio_hash]['genre'] = genre
            self.analysis_cache[audio_hash]['confidence'] = confidence
            
            return genre, confidence
        except Exception as e:
            print(f"Error during genre classification: {e}")
            # Fallback to simple heuristic in case of error
            return self._dummy_classify(audio_data)
    
    def _dummy_classify(self, audio_data):
        """Fallback classification method using simple heuristics"""
        # Extract basic features
        features = self.extract_features(audio_data)
        
        # Simple heuristic for demonstration purposes
        tempo = features["tempo"]
        spectral_centroid_mean = np.mean(features["spectral_centroid"]) if len(features["spectral_centroid"]) > 0 else 0
        
        # These are arbitrary thresholds for demonstration
        if spectral_centroid_mean > 2000 and tempo > 130:
            genre_idx = self.genres.index('disco')  # or closest match
            confidence = 85.5
        elif spectral_centroid_mean > 1800 and tempo > 100:
            genre_idx = self.genres.index('pop')
            confidence = 78.3
        elif spectral_centroid_mean < 1200 and tempo < 90:
            genre_idx = self.genres.index('classical')
            confidence = 82.7
        elif 1400 < spectral_centroid_mean < 1800 and 80 < tempo < 120:
            genre_idx = self.genres.index('jazz')
            confidence = 76.9
        elif spectral_centroid_mean > 1800 and tempo > 120:
            genre_idx = self.genres.index('metal')
            confidence = 88.2
        else:
            # Default to rock with medium confidence
            genre_idx = self.genres.index('rock')
            confidence = 65.0
            
        return self.genres[genre_idx], confidence
        
    def analyze(self, audio_data, debug_instruments=False):
        """
        Analyze audio data to classify genre and extract components
        
        Args:
            audio_data: Numpy array of audio samples
            debug_instruments: Whether to print debug information for instrument detection
            
        Returns:
            Tuple of (genre, confidence, components)
        """
        start_time = time.time()
        
        # Use first 30 seconds of audio for full analysis if available
        max_length = min(len(audio_data), self.sample_rate * 30)
        audio_segment = audio_data[:max_length]
        
        # Check if complete analysis is in cache
        audio_hash = hash(audio_segment.tobytes())
        if audio_hash in self.analysis_cache and 'components' in self.analysis_cache[audio_hash] and not debug_instruments:
            genre = self.analysis_cache[audio_hash]['genre']
            confidence = self.analysis_cache[audio_hash]['confidence']
            components = self.analysis_cache[audio_hash]['components']
            # print(f"Using cached complete analysis ({time.time() - start_time:.3f}s)")
            return genre, confidence, components
        
        # Extract features for component analysis
        features = self.extract_features(audio_segment)
        
        # Classify genre directly from audio data
        genre, confidence = self.classify_genre(audio_segment)
        
        # Analyze components
        components = self.analyze_components(features)
        
        # Detect instruments - new addition
        instruments = self.instrument_detector.detect_instruments(
            audio_segment, 
            self.sample_rate, 
            genre, 
            debug=debug_instruments
        )
        components['instruments'] = instruments
        
        # Cache the complete analysis
        if audio_hash not in self.analysis_cache:
            self.analysis_cache[audio_hash] = {}
        self.analysis_cache[audio_hash]['genre'] = genre
        self.analysis_cache[audio_hash]['confidence'] = confidence
        self.analysis_cache[audio_hash]['components'] = components
        
        # Remove oldest entries if cache gets too large
        if len(self.analysis_cache) > self.cache_size_limit:
            oldest_key = next(iter(self.analysis_cache))
            del self.analysis_cache[oldest_key]
        
        end_time = time.time()
        print(f"Audio analysis completed in {end_time - start_time:.3f} seconds")
        
        return genre, confidence, components
        
    def get_component_details(self, component_type, click_data):
        """
        Get detailed descriptions for a specific component based on click data
        
        Args:
            component_type: Type of component ('rhythm', 'melody', 'instrumentation')
            click_data: Data from click event on visualization
            
        Returns:
            Dash HTML content with detailed description
        """
        # Extract relevant information from click_data
        if not click_data or 'points' not in click_data or not click_data['points']:
            return "No data available for this component."
            
        point = click_data['points'][0]
        
        # Get feature name from point
        if 'customdata' in point and point['customdata'] is not None:
            feature = point['customdata']
        else:
            # Use the most relevant feature based on the component type
            if component_type == 'rhythm':
                feature = 'tempo'
            elif component_type == 'melody':
                feature = 'melodic_contour'
            else:  # instrumentation
                feature = 'timbre'
                
        # Get descriptions
        general_desc = self.component_descriptions[component_type].get(feature, "")
        
        # For context related to the current genre, provide genre-specific descriptions
        # This would be enhanced in a real implementation to show the most relevant genre
        genre_examples = []
        example_genres = [g for g in self.genres if g in self.component_descriptions[component_type]]
        
        if example_genres:
            # Take up to 3 genre examples
            for genre in example_genres[:3]:
                if genre in self.component_descriptions[component_type]:
                    genre_examples.append(
                        html.P([
                            html.Strong(f"{genre.capitalize()}: "), 
                            self.component_descriptions[component_type][genre]
                        ])
                    )
                    
        # Construct HTML content
        content = [
            html.H4(feature.replace('_', ' ').title()),
            html.P(general_desc),
            html.H5("Genre Examples:")
        ]
        
        # Add genre examples if available
        if genre_examples:
            content.extend(genre_examples)
        else:
            content.append(html.P("No specific examples available."))
        
        return html.Div(content)

    def get_genre_explanation(self, genre, components):
        """
        Get detailed explanation of why audio was classified as a specific genre
        
        Args:
            genre: The classified genre
            components: Dictionary of analyzed components (rhythm, melody, instrumentation)
            
        Returns:
            Explanation text formatted for Dash
        """
        # If genre is not in our descriptions or not a valid genre, return basic message
        if genre not in self.genre_descriptions:
            return [
                html.P([f"The audio was classified as ", html.Strong(genre), ", but detailed explanation is not available for this genre."])
            ]
        
        # Get genre description
        genre_info = self.genre_descriptions[genre]
        
        # Create basic explanation
        explanation = [
            html.H4(f"Why was this classified as {genre.capitalize()}?"),
            html.P(genre_info['description']),
            
            html.H5("Key Characteristics:"),
            html.Ul([
                html.Li([html.Strong("Rhythm:"), f" {genre_info['rhythm']}"]),
                html.Li([html.Strong("Melody:"), f" {genre_info['melody']}"]),
                html.Li([html.Strong("Instrumentation:"), f" {genre_info['instrumentation']}"])
            ])
        ]
        
        # Add specific component matches if available
        features_list = []
        
        if 'rhythm' in components and components['rhythm']:
            rhythm = components['rhythm']
            tempo = rhythm.get('tempo', 0)
            tempo_category = rhythm.get('tempo_category', 'Unknown')
            complexity = rhythm.get('complexity_category', 'Unknown')
            
            if genre == 'disco' and tempo >= 110 and tempo <= 130:
                features_list.append(html.Li(f"Tempo of {tempo:.1f} BPM matches typical disco range (110-130 BPM)"))
            elif genre == 'metal' and tempo >= 100:
                features_list.append(html.Li(f"Faster tempo of {tempo:.1f} BPM aligns with metal's driving rhythms"))
            elif genre == 'blues' and tempo >= 60 and tempo <= 100:
                features_list.append(html.Li(f"Moderate tempo of {tempo:.1f} BPM is characteristic of blues"))
            else:
                features_list.append(html.Li(f"Tempo: {tempo:.1f} BPM ({tempo_category})"))
                
            features_list.append(html.Li(f"Rhythm complexity: {complexity}"))
            
        if 'melody' in components and components['melody']:
            melody = components['melody']
            modality = melody.get('modality', 'Unknown')
            variety = melody.get('variety_category', 'Unknown')
            dominant_notes = ', '.join(melody.get('dominant_notes', ['Unknown']))
            
            features_list.append(html.Li(f"Melodic key/mode: {modality}"))
            
            if genre == 'jazz' and variety == 'High':
                features_list.append(html.Li("High pitch variety is typical of jazz's complex melodic structure"))
            elif genre == 'pop' and variety == 'Medium':
                features_list.append(html.Li("Medium pitch variety aligns with pop's accessible melodic approach"))
            else:
                features_list.append(html.Li(f"Pitch variety: {variety}"))
                
            features_list.append(html.Li(f"Dominant notes: {dominant_notes}"))
            
        if 'instrumentation' in components and components['instrumentation']:
            instrumentation = components['instrumentation']
            brightness = instrumentation.get('brightness_category', 'Unknown')
            complexity = instrumentation.get('complexity_category', 'Unknown')
            
            if genre == 'metal' and brightness == 'Bright/Sharp':
                features_list.append(html.Li("Bright/sharp timbre consistent with metal's distorted guitar sound"))
            elif genre == 'classical' and complexity == 'Complex/Rich':
                features_list.append(html.Li("Complex/rich timbral texture matches classical orchestration"))
            else:
                features_list.append(html.Li(f"Timbral brightness: {brightness}"))
                features_list.append(html.Li(f"Timbral complexity: {complexity}"))
        
        # Add detected features section if we have any features
        if features_list:
            explanation.extend([
                html.H5("Detected Features:"),
                html.Ul(features_list)
            ])
            
        return html.Div(explanation, className="genre-explanation")
        
    def get_available_genres(self):
        """
        Get information about available genres in the model
        
        Returns:
            Content formatted for Dash
        """
        genre_cards = []
        
        for genre in self.genres:
            description = self.genre_descriptions.get(genre, {}).get('description', 'No description available.')
            genre_cards.append(
                html.Div([
                    html.H5(genre.capitalize()),
                    html.P(description)
                ], className="genre-card")
            )
            
        return html.Div([
            html.H4("Genres Available in Our Model"),
            html.P("The current model can classify audio into the following 10 genres:"),
            html.Div(genre_cards, className="genre-grid")
        ], className="available-genres")

    def get_instrument_details(self, instruments, genre):
        """
        Format instrument detection results for display
        
        Args:
            instruments: List of detected instruments with their properties
            genre: The detected genre
            
        Returns:
            Dash HTML for displaying instrument information
        """
        if not instruments or len(instruments) == 0:
            return html.Div([
                html.P("No instruments were confidently detected in this audio.")
            ], className="instrument-details")
            
        # Create content for each detected instrument
        instrument_items = []
        for instrument in instruments:
            instrument_items.append(html.Div([
                html.H5([
                    instrument['name'], 
                    html.Span(f" ({instrument['confidence']:.1f}%)", 
                             className="confidence-score")
                ]),
                html.P(instrument['description']),
                html.P([
                    html.Strong("Role in music: "), 
                    instrument['role']
                ])
            ], className="instrument-item"))
            
        # Assemble the complete content
        content = [
            html.H4(f"Detected Instruments in {genre.capitalize()}"),
            html.P("The following instruments were identified in the audio recording:"),
            html.Div(instrument_items, className="instrument-list")
        ]
        
        return html.Div(content, className="instrument-details") 