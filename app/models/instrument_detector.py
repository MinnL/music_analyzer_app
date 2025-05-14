import numpy as np
import librosa
import torch
import time

class InstrumentDetector:
    """
    A class for detecting musical instruments in audio recordings
    """
    
    def __init__(self):
        """
        Initialize the instrument detector
        """
        # Dictionary of instruments with their spectral characteristics
        self.instruments = {
            'piano': {
                'description': 'A keyboard instrument that produces sound by striking strings with hammers.',
                'spectral_range': (30, 4200),  # Hz
                'attack_time': 'short',
                'decay_profile': 'medium',
                'harmonic_profile': 'rich',
                'feature_weights': {
                    'spectral_centroid': 0.7,
                    'spectral_bandwidth': 0.6, 
                    'spectral_contrast': 0.7,
                    'spectral_rolloff': 0.5,
                    'zero_crossing_rate': 0.3,
                    'piano_ratio': 0.9,
                    'percussion_ratio': 0.3,
                    'onset_frequency': 0.6,
                    'percussive_energy_ratio': 0.4
                }
            },
            'acoustic_guitar': {
                'description': 'A string instrument that produces sound by plucking or strumming strings.',
                'spectral_range': (80, 5000),  # Hz
                'attack_time': 'medium',
                'decay_profile': 'long',
                'harmonic_profile': 'rich',
                'feature_weights': {
                    'spectral_centroid': 0.6,
                    'spectral_bandwidth': 0.5,
                    'spectral_contrast': 0.8,
                    'spectral_rolloff': 0.6,
                    'zero_crossing_rate': 0.4
                }
            },
            'electric_guitar': {
                'description': 'An electric string instrument that produces sound through magnetic pickups and amplification.',
                'spectral_range': (80, 6000),  # Hz
                'attack_time': 'medium',
                'decay_profile': 'variable',
                'harmonic_profile': 'rich_distorted',
                'feature_weights': {
                    'spectral_centroid': 0.7,
                    'spectral_bandwidth': 0.7,
                    'spectral_contrast': 0.9,
                    'spectral_rolloff': 0.8,
                    'zero_crossing_rate': 0.6
                }
            },
            'violin': {
                'description': 'A bowed string instrument with a high pitch range.',
                'spectral_range': (200, 3500),  # Hz
                'attack_time': 'variable',
                'decay_profile': 'sustained',
                'harmonic_profile': 'rich',
                'feature_weights': {
                    'spectral_centroid': 0.8,
                    'spectral_bandwidth': 0.5,
                    'spectral_contrast': 0.7,
                    'spectral_rolloff': 0.6,
                    'zero_crossing_rate': 0.5
                }
            },
            'cello': {
                'description': 'A bowed string instrument with a low to middle pitch range.',
                'spectral_range': (65, 1000),  # Hz
                'attack_time': 'variable',
                'decay_profile': 'sustained',
                'harmonic_profile': 'rich_resonant',
                'feature_weights': {
                    'spectral_centroid': 0.4,
                    'spectral_bandwidth': 0.6,
                    'spectral_contrast': 0.7,
                    'spectral_rolloff': 0.4,
                    'zero_crossing_rate': 0.3
                }
            },
            'trumpet': {
                'description': 'A brass instrument with a bright, piercing sound.',
                'spectral_range': (160, 1000),  # Hz
                'attack_time': 'short',
                'decay_profile': 'sustained',
                'harmonic_profile': 'bright_brassy',
                'feature_weights': {
                    'spectral_centroid': 0.9,
                    'spectral_bandwidth': 0.7,
                    'spectral_contrast': 0.6,
                    'spectral_rolloff': 0.8,
                    'zero_crossing_rate': 0.5
                }
            },
            'saxophone': {
                'description': 'A woodwind instrument with a distinctive reedy sound.',
                'spectral_range': (100, 3000),  # Hz
                'attack_time': 'medium',
                'decay_profile': 'sustained',
                'harmonic_profile': 'rich_reedy',
                'feature_weights': {
                    'spectral_centroid': 0.7,
                    'spectral_bandwidth': 0.8,
                    'spectral_contrast': 0.7,
                    'spectral_rolloff': 0.6,
                    'zero_crossing_rate': 0.5
                }
            },
            'drums': {
                'description': 'Percussion instruments including kicks, snares, hi-hats, and cymbals.',
                'spectral_range': (40, 10000),  # Hz
                'attack_time': 'very_short',
                'decay_profile': 'variable',
                'harmonic_profile': 'noisy',
                'feature_weights': {
                    'spectral_centroid': 0.6,
                    'spectral_bandwidth': 0.9,
                    'spectral_contrast': 0.9,
                    'spectral_rolloff': 0.8,
                    'zero_crossing_rate': 0.9,
                    'percussion_ratio': 0.95,
                    'onset_frequency': 0.85,
                    'piano_ratio': 0.1,
                    'percussive_energy_ratio': 0.95
                }
            },
            'bass': {
                'description': 'A low-pitched string instrument, either acoustic or electric.',
                'spectral_range': (40, 400),  # Hz
                'attack_time': 'medium',
                'decay_profile': 'variable',
                'harmonic_profile': 'fundamental_heavy',
                'feature_weights': {
                    'spectral_centroid': 0.2,
                    'spectral_bandwidth': 0.5,
                    'spectral_contrast': 0.6,
                    'spectral_rolloff': 0.3,
                    'zero_crossing_rate': 0.2
                }
            },
            'synthesizer': {
                'description': 'An electronic instrument that generates sounds through various synthesis methods.',
                'spectral_range': (20, 20000),  # Hz (wide range)
                'attack_time': 'variable',
                'decay_profile': 'variable',
                'harmonic_profile': 'variable',
                'feature_weights': {
                    'spectral_centroid': 0.7,
                    'spectral_bandwidth': 0.8,
                    'spectral_contrast': 0.7,
                    'spectral_rolloff': 0.7,
                    'zero_crossing_rate': 0.6
                }
            }
        }
        
        # Instrument roles in different genres
        self.instrument_roles = {
            'piano': {
                'classical': 'Often serves as a solo instrument or provides harmonic structure in chamber music.',
                'jazz': 'Provides harmonic foundation and is often used for improvised solos.',
                'blues': 'Adds rhythmic and harmonic support, often with distinctive walking bass lines.',
                'pop': 'Typically provides melodic or harmonic elements, sometimes as the main instrument.'
            },
            'acoustic_guitar': {
                'country': 'Forms the backbone of the genre, often with fingerpicking patterns or strumming.',
                'folk': 'Primary instrument that carries both rhythm and melody.',
                'rock': 'Often used for ballads or softer sections, providing texture and warmth.',
                'pop': 'Frequently used to add organic texture and warmth to predominantly electronic arrangements.'
            },
            'electric_guitar': {
                'rock': 'Defines the genre with power chords, riffs, and solos.',
                'metal': 'The central instrument, often heavily distorted with technical solos.',
                'blues': 'Delivers emotional solos and characteristic bends and vibrato.',
                'jazz': 'Used in fusion styles, offering clean tones with complex chord voicings.'
            },
            'violin': {
                'classical': 'Essential orchestral instrument that often carries the main melody or leads the string section.',
                'folk': 'Provides melodies, counter-melodies, and rhythmic elements in traditional folk music.',
                'country': 'Known as a fiddle in this context, adds distinctive melodic lines and solos.'
            },
            'cello': {
                'classical': 'Provides bass lines in orchestral music and has a prominent role in chamber music.',
                'pop': 'Occasionally used to add depth and emotional resonance to ballads or orchestral pop.'
            },
            'trumpet': {
                'jazz': 'Iconic melodic and solo instrument, especially in big band and bebop styles.',
                'classical': 'Used for fanfares, dramatic moments, and melodic lines in orchestral music.',
                'ska': 'Part of the horn section that defines the genre\'s characteristic upbeat sound.'
            },
            'saxophone': {
                'jazz': 'Quintessential jazz instrument used for expressive improvisation and melody.',
                'blues': 'Adds soulful melodic lines and improvisations.',
                'rock': 'Occasionally featured for solos and melodic lines, especially in classic rock.'
            },
            'drums': {
                'rock': 'Provides the driving beat and energy that defines rock music.',
                'jazz': 'Maintains time while adding complex rhythmic interplay and improvisation.',
                'hiphop': 'The foundation of beats, either sampled or programmed.',
                'metal': 'Features aggressive, often double-bass patterns that drive the intensity.'
            },
            'bass': {
                'funk': 'Takes a lead role with slap techniques and rhythmic complexity.',
                'rock': 'Works with drums to form the rhythmic foundation.',
                'jazz': 'Provides walking bass lines that outline harmonies and structure.',
                'reggae': 'Plays a dominant role with distinctive, prominent bass lines.'
            },
            'synthesizer': {
                'electronic': 'The primary instrument creating both melodic and textural elements.',
                'pop': 'Used for pads, leads, and sound effects in contemporary production.',
                'disco': 'Creates distinctive arpeggios and bass lines.',
                'hiphop': 'Used for melodic hooks and atmospheric elements in modern production.'
            }
        }
        
    def extract_features(self, audio_data, sr):
        """
        Extract audio features for instrument detection
        
        Args:
            audio_data: Numpy array of audio samples
            sr: Sample rate
            
        Returns:
            Dictionary of audio features
        """
        # Use shorter audio segment (max 10 seconds) for faster processing
        max_length = min(len(audio_data), sr * 10)
        audio_segment = audio_data[:max_length]
        
        # Compute various spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sr).mean()
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_segment, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_segment).mean()
        
        # Enhanced onset detection for percussion and piano
        # Higher hop length makes detection more sensitive to more prominent onsets
        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=audio_segment, sr=sr, hop_length=hop_length)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        
        # Calculate onset frequency - helps identify percussive instruments like drums
        onset_frequency = len(onset_frames) / (len(audio_segment) / sr)
        
        # Calculate average attack time if onsets are detected
        if len(onset_frames) > 1:
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
            attack_times = []
            
            # Attack patterns for different instruments
            percussive_count = 0
            piano_like_count = 0
            
            for i in range(len(onset_frames) - 1):
                start_frame = onset_frames[i]
                # Look at a short window after onset
                end_frame = min(start_frame + 5, len(onset_env) - 1)
                
                if start_frame < end_frame:
                    # Calculate how quickly the sound reaches peak
                    segment = onset_env[start_frame:end_frame]
                    if len(segment) > 0:
                        rise_time = np.argmax(segment)
                        attack_times.append(rise_time)
                        
                        # Identify percussive onsets (very quick attacks, like drums)
                        if rise_time <= 1:
                            percussive_count += 1
                            
                        # Identify piano-like onsets (quick attack but not immediate)
                        if rise_time == 1 or rise_time == 2:
                            piano_like_count += 1
                        
            avg_attack = np.mean(attack_times) if attack_times else 3  # Default middle value
            
            # Calculate ratios to identify instrument types
            percussion_ratio = percussive_count / len(onset_frames) if len(onset_frames) > 0 else 0
            piano_ratio = piano_like_count / len(onset_frames) if len(onset_frames) > 0 else 0
        else:
            avg_attack = 3  # Default middle value
            percussion_ratio = 0
            piano_ratio = 0
            onset_frequency = 0
            
        # Calculate harmonic-percussive separation to help identify drums vs. tonal instruments
        y_harmonic, y_percussive = librosa.effects.hpss(audio_segment)
        harmonic_energy = np.mean(y_harmonic**2)
        percussive_energy = np.mean(y_percussive**2)
        
        # Ratio of percussive to total energy
        if (harmonic_energy + percussive_energy) > 0:
            percussive_ratio = percussive_energy / (harmonic_energy + percussive_energy)
        else:
            percussive_ratio = 0
            
        # Normalize features to 0-1 range for easier comparison
        max_centroid = sr / 2  # Nyquist frequency
        norm_centroid = float(spectral_centroid / max_centroid)
        norm_bandwidth = float(spectral_bandwidth / max_centroid)
        norm_rolloff = float(spectral_rolloff / max_centroid)
        
        # Create a consolidated feature dictionary
        features = {
            'spectral_centroid': norm_centroid,
            'spectral_bandwidth': norm_bandwidth,
            'spectral_contrast': float(spectral_contrast),
            'spectral_rolloff': norm_rolloff,
            'zero_crossing_rate': float(zero_crossing_rate),
            'attack_time': float(avg_attack),
            'percussion_ratio': float(percussion_ratio),
            'piano_ratio': float(piano_ratio),
            'onset_frequency': float(onset_frequency),
            'percussive_energy_ratio': float(percussive_ratio)
        }
        
        return features
        
    def detect_instruments(self, audio_data, sr, genre=None, confidence_threshold=0.45, debug=False):
        """
        Detect instruments in audio data
        
        Args:
            audio_data: Numpy array of audio samples
            sr: Sample rate
            genre: Optionally provide detected genre to improve accuracy
            confidence_threshold: Minimum confidence to include an instrument in results
            debug: Whether to print debug information
            
        Returns:
            List of tuples (instrument, confidence, role_in_genre)
        """
        start_time = time.time()
        
        # Extract features from audio
        features = self.extract_features(audio_data, sr)
        
        if debug:
            print("Extracted features:")
            for feature, value in features.items():
                print(f"  {feature}: {value:.4f}")
        
        # Calculate similarity scores for each instrument
        instrument_scores = {}
        
        # List of instruments to always consider regardless of genre
        always_consider = ['piano', 'drums']
        
        for instrument, specs in self.instruments.items():
            # Skip certain instruments based on genre if provided
            if genre and instrument not in always_consider:
                # For instance, skip synthesizers if genre is classical
                if genre == "classical" and instrument in ["synthesizer", "electric_guitar"]:
                    continue
                # Skip orchestral instruments in electronic genres
                if genre in ["electronic", "hiphop", "disco"] and instrument in ["violin", "cello", "trumpet"]:
                    continue
            
            # Calculate weighted score based on feature similarity
            score = 0
            weights_sum = 0
            
            if debug and instrument in ['piano', 'drums']:
                print(f"\nFeature scores for {instrument}:")
            
            for feature, weight in specs['feature_weights'].items():
                if feature in features:
                    # Feature-specific comparison logic
                    if feature == 'spectral_centroid':
                        # Calculate how well the centroid matches the instrument's expected range
                        ideal_centroid = (specs['spectral_range'][0] + specs['spectral_range'][1]) / (2 * sr/2)
                        feature_score = 1 - min(abs(features[feature] - ideal_centroid) * 2, 1)
                    elif feature == 'percussion_ratio':
                        # For drums, higher is better
                        if instrument == 'drums':
                            feature_score = features[feature]
                        else:
                            # For non-percussion, lower values are expected
                            feature_score = 1 - features[feature]
                    elif feature == 'piano_ratio':
                        # Direct match for piano
                        if instrument == 'piano':
                            feature_score = features[feature]
                        else:
                            # For non-piano, this is less important
                            feature_score = 0.5  # Neutral score
                    elif feature == 'onset_frequency':
                        # Drums have high onset frequency
                        if instrument == 'drums':
                            # Higher is better for drums (up to a point)
                            feature_score = min(features[feature] * 2, 1.0)
                        elif instrument == 'piano':
                            # Piano has moderate onset frequency
                            optimal_freq = 0.5  # Moderate onset frequency
                            feature_score = 1 - min(abs(features[feature] - optimal_freq) * 2, 1)
                        else:
                            # Generic handling for other instruments
                            feature_score = 1 - min(abs(features[feature] - 0.3) * 2, 1)
                    elif feature == 'percussive_energy_ratio':
                        # Drums have high percussive energy
                        if instrument == 'drums':
                            feature_score = features[feature]
                        elif instrument in ['piano', 'acoustic_guitar']:
                            # These have moderate percussive energy
                            optimal_ratio = 0.4
                            feature_score = 1 - min(abs(features[feature] - optimal_ratio) * 2, 1)
                        else:
                            # Other instruments have low percussive energy
                            feature_score = 1 - features[feature]
                    else:
                        # For other features, use a simpler scoring approach
                        feature_score = 1 - min(abs(features[feature] - 0.5) * 1.5, 1)
                    
                    original_score = feature_score
                    
                    # Enhanced detection for piano
                    if instrument == 'piano' and feature == 'spectral_contrast':
                        # Pianos have distinctive spectral contrast from harmonic structure
                        feature_score *= 1.2
                        feature_score = min(feature_score, 1.0)
                    
                    # Enhanced detection for drums
                    if instrument == 'drums' and feature == 'zero_crossing_rate':
                        # Drums have distinctive zero crossing rates
                        feature_score *= 1.3
                        feature_score = min(feature_score, 1.0)
                    
                    if debug and instrument in ['piano', 'drums']:
                        if original_score != feature_score:
                            print(f"  {feature}: {original_score:.4f} -> {feature_score:.4f} (weight: {weight:.2f})")
                        else:
                            print(f"  {feature}: {feature_score:.4f} (weight: {weight:.2f})")
                    
                    score += feature_score * weight
                    weights_sum += weight
            
            # Normalize score
            if weights_sum > 0:
                normalized_score = score / weights_sum
                
                if debug and instrument in ['piano', 'drums']:
                    print(f"Raw score for {instrument}: {normalized_score:.4f}")
                
                # Apply genre-based boosting if genre is provided
                original_score = normalized_score
                
                if genre and instrument in self.instrument_roles and genre in self.instrument_roles[instrument]:
                    # Boost instruments commonly found in this genre
                    normalized_score *= 1.2
                    normalized_score = min(normalized_score, 1.0)  # Cap at 1.0
                
                # Special boosting for piano in classical and jazz
                if instrument == 'piano' and genre in ['classical', 'jazz']:
                    normalized_score *= 1.15
                    normalized_score = min(normalized_score, 1.0)
                
                # Special boosting for drums in rock, pop, and jazz
                if instrument == 'drums' and genre in ['rock', 'pop', 'jazz']:
                    normalized_score *= 1.15
                    normalized_score = min(normalized_score, 1.0)
                
                if debug and instrument in ['piano', 'drums'] and original_score != normalized_score:
                    print(f"Adjusted score for {instrument}: {original_score:.4f} -> {normalized_score:.4f}")
                
                instrument_scores[instrument] = normalized_score
        
        # Sort instruments by score and filter by threshold
        sorted_instruments = sorted(
            [(i, s) for i, s in instrument_scores.items() if s >= confidence_threshold],
            key=lambda x: x[1],
            reverse=True
        )
        
        if debug:
            print("\nAll instrument scores:")
            for instrument, score in sorted(instrument_scores.items(), key=lambda x: x[1], reverse=True):
                threshold_info = " (below threshold)" if score < confidence_threshold else ""
                print(f"  {instrument}: {score:.4f}{threshold_info}")
        
        # Cap at top 5 instruments
        top_instruments = sorted_instruments[:5]
        
        # Add genre-specific role information if genre is provided
        results = []
        for instrument, score in top_instruments:
            role = "N/A"
            if genre and instrument in self.instrument_roles and genre in self.instrument_roles[instrument]:
                role = self.instrument_roles[instrument][genre]
            elif instrument in self.instrument_roles:
                # Fall back to a common role if specific genre role isn't available
                common_role = next(iter(self.instrument_roles[instrument].values()), "N/A")
                role = common_role
                
            # Add instrument description
            description = self.instruments[instrument]['description']
            
            results.append({
                'name': instrument.replace('_', ' ').title(),
                'confidence': float(score * 100),  # Convert to percentage
                'description': description,
                'role': role
            })
        
        print(f"Instrument detection completed in {time.time() - start_time:.3f} seconds")
        return results 