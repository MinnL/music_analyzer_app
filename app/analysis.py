import numpy as np
import librosa
import torch
import torchaudio
from sklearn.preprocessing import StandardScaler
import os
import json

class MusicAnalyzer:
    """
    Class for analyzing audio and classifying genres
    """
    
    def __init__(self, model_path=None, sample_rate=22050):
        """
        Initialize the music analyzer
        
        Args:
            model_path: Path to pre-trained model (if None, a dummy model will be used)
            sample_rate: Sample rate for audio analysis
        """
        self.sample_rate = sample_rate
        self.model = None
        self.genres = ["classical", "country", "electronic", "hip-hop", "jazz", 
                      "metal", "pop", "reggae", "rock", "blues"]
        
        # Component descriptions for various genres and audio features
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
        
        # Load or initialize model
        self._load_or_init_model(model_path)
        
    def _load_or_init_model(self, model_path):
        """Load a pre-trained model or initialize a dummy model for demonstration"""
        if model_path and os.path.exists(model_path):
            # Load actual model (for a real implementation)
            try:
                # This is a placeholder for actual model loading code
                pass
            except Exception as e:
                print(f"Error loading model: {e}")
                self._init_dummy_model()
        else:
            # For demonstration, use a dummy model
            self._init_dummy_model()
            
    def _init_dummy_model(self):
        """Initialize a dummy model for demonstration purposes"""
        # This would be replaced with actual model initialization in a real implementation
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
                "tempo": 0,
                "spectral_centroid": np.array([]),
                "spectral_rolloff": np.array([]),
                "spectral_contrast": np.array([]),
                "chroma": np.array([]),
                "mfcc": np.array([])
            }
            
        # Compute basic audio features using librosa
        try:
            # Tempo and beat information
            onset_env = librosa.onset.onset_strength(y=audio_data, sr=self.sample_rate)
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=self.sample_rate)
            
            # Tonal features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
            
            # Timbre features
            mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            
            return {
                "tempo": tempo,
                "spectral_centroid": spectral_centroid,
                "spectral_rolloff": spectral_rolloff,
                "spectral_contrast": spectral_contrast,
                "chroma": chroma,
                "mfcc": mfcc
            }
        except Exception as e:
            print(f"Error extracting features: {e}")
            return {
                "tempo": 0,
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
        # This is a simplified analysis for demonstration
        # In a real implementation, this would involve more sophisticated algorithms
        
        # Analyze rhythm
        rhythm = {}
        if features["tempo"] > 0:
            rhythm["tempo"] = features["tempo"]
            
            # Categorize tempo
            if features["tempo"] < 70:
                rhythm["tempo_category"] = "Slow"
            elif features["tempo"] < 120:
                rhythm["tempo_category"] = "Medium"
            else:
                rhythm["tempo_category"] = "Fast"
                
            # Simple rhythm complexity estimation based on spectral contrast variation
            if len(features["spectral_contrast"]) > 0:
                rhythm_complexity = np.mean(np.std(features["spectral_contrast"], axis=1))
                rhythm["complexity"] = rhythm_complexity
                
                if rhythm_complexity < 0.4:
                    rhythm["complexity_category"] = "Simple"
                elif rhythm_complexity < 0.8:
                    rhythm["complexity_category"] = "Moderate"
                else:
                    rhythm["complexity_category"] = "Complex"
        else:
            rhythm = {"tempo": 0, "tempo_category": "Unknown", "complexity": 0, "complexity_category": "Unknown"}
            
        # Analyze melody using chroma features
        melody = {}
        if features["chroma"].size > 0:
            # Dominant pitch classes
            chroma_mean = np.mean(features["chroma"], axis=1)
            dominant_notes = np.argsort(-chroma_mean)[:3]  # Top 3 dominant notes
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            melody["dominant_notes"] = [note_names[i % 12] for i in dominant_notes]
            
            # Pitch variety
            pitch_variety = np.std(chroma_mean)
            melody["pitch_variety"] = pitch_variety
            
            if pitch_variety < 0.1:
                melody["variety_category"] = "Low"
            elif pitch_variety < 0.2:
                melody["variety_category"] = "Medium"
            else:
                melody["variety_category"] = "High"
                
            # Estimate if major or minor based on relative presence of major/minor thirds
            major_third_idx = (dominant_notes[0] + 4) % 12
            minor_third_idx = (dominant_notes[0] + 3) % 12
            
            if chroma_mean[major_third_idx] > chroma_mean[minor_third_idx]:
                melody["modality"] = "Major"
            else:
                melody["modality"] = "Minor"
        else:
            melody = {
                "dominant_notes": [], 
                "pitch_variety": 0, 
                "variety_category": "Unknown",
                "modality": "Unknown"
            }
            
        # Analyze instrumentation using MFCCs and spectral features
        instrumentation = {}
        if features["mfcc"].size > 0:
            # Spectral centroid correlates with brightness/sharpness
            if features["spectral_centroid"].size > 0:
                brightness = np.mean(features["spectral_centroid"]) / (self.sample_rate/2)  # Normalize to 0-1
                instrumentation["brightness"] = brightness
                
                if brightness < 0.3:
                    instrumentation["brightness_category"] = "Dark/Warm"
                elif brightness < 0.6:
                    instrumentation["brightness_category"] = "Balanced"
                else:
                    instrumentation["brightness_category"] = "Bright/Sharp"
            else:
                instrumentation["brightness"] = 0
                instrumentation["brightness_category"] = "Unknown"
                
            # Spectral contrast correlates with instrument separation/clarity
            if features["spectral_contrast"].size > 0:
                contrast = np.mean(np.mean(features["spectral_contrast"]))
                instrumentation["contrast"] = contrast
                
                if contrast < 20:
                    instrumentation["contrast_category"] = "Blended/Smooth"
                elif contrast < 40:
                    instrumentation["contrast_category"] = "Balanced"
                else:
                    instrumentation["contrast_category"] = "Distinct/Clear"
            else:
                instrumentation["contrast"] = 0
                instrumentation["contrast_category"] = "Unknown"
                
            # Overall timbre complexity from MFCC variance
            timbre_complexity = np.mean(np.std(features["mfcc"], axis=1))
            instrumentation["timbre_complexity"] = timbre_complexity
            
            if timbre_complexity < 10:
                instrumentation["complexity_category"] = "Simple/Clean"
            elif timbre_complexity < 30:
                instrumentation["complexity_category"] = "Moderate"
            else:
                instrumentation["complexity_category"] = "Complex/Rich"
        else:
            instrumentation = {
                "brightness": 0, 
                "brightness_category": "Unknown",
                "contrast": 0,
                "contrast_category": "Unknown",
                "timbre_complexity": 0,
                "complexity_category": "Unknown"
            }
            
        return {
            "rhythm": rhythm,
            "melody": melody,
            "instrumentation": instrumentation
        }
        
    def classify_genre(self, features):
        """
        Classify music genre based on audio features
        
        Args:
            features: Dictionary of audio features
            
        Returns:
            Tuple of (genre, confidence)
        """
        # This is a dummy implementation for demonstration
        # In a real implementation, this would use the actual trained model
        
        # If features are empty, return unknown
        if len(features["spectral_centroid"]) == 0:
            return "Unknown", 0.0
            
        # Simple heuristic for demonstration purposes
        tempo = features["tempo"]
        spectral_centroid_mean = np.mean(features["spectral_centroid"]) if len(features["spectral_centroid"]) > 0 else 0
        
        # These are arbitrary thresholds for demonstration
        if spectral_centroid_mean > 2000 and tempo > 130:
            genre_idx = 2  # electronic
            confidence = 85.5
        elif spectral_centroid_mean > 1800 and tempo > 100:
            genre_idx = 6  # pop
            confidence = 78.3
        elif spectral_centroid_mean < 1200 and tempo < 90:
            genre_idx = 0  # classical
            confidence = 82.7
        elif 1400 < spectral_centroid_mean < 1800 and 80 < tempo < 120:
            genre_idx = 4  # jazz
            confidence = 76.9
        elif spectral_centroid_mean > 1800 and tempo > 120:
            genre_idx = 5  # metal
            confidence = 88.2
        else:
            # Default to rock with medium confidence
            genre_idx = 8  # rock
            confidence = 65.0
            
        return self.genres[genre_idx], confidence
        
    def analyze(self, audio_data):
        """
        Analyze audio data to classify genre and extract components
        
        Args:
            audio_data: Numpy array of audio samples
            
        Returns:
            Tuple of (genre, confidence, components)
        """
        # Extract features
        features = self.extract_features(audio_data)
        
        # Classify genre
        genre, confidence = self.classify_genre(features)
        
        # Analyze components
        components = self.analyze_components(features)
        
        return genre, confidence, components
        
    def get_component_details(self, component_type, click_data):
        """
        Get detailed descriptions for a specific component based on click data
        
        Args:
            component_type: Type of component ('rhythm', 'melody', 'instrumentation')
            click_data: Data from click event on visualization
            
        Returns:
            HTML content with detailed description
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
        genre_desc = ""
        genre_examples = [g for g in self.genres if g in self.component_descriptions[component_type]]
        
        if genre_examples:
            # Take up to 3 genre examples
            for genre in genre_examples[:3]:
                if genre in self.component_descriptions[component_type]:
                    genre_desc += f"<p><strong>{genre.capitalize()}:</strong> {self.component_descriptions[component_type][genre]}</p>"
                    
        # Construct HTML content
        html_content = f"""
        <div>
            <h4>{feature.replace('_', ' ').title()}</h4>
            <p>{general_desc}</p>
            <h5>Genre Examples:</h5>
            {genre_desc}
        </div>
        """
        
        return html_content 