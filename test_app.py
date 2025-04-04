#!/usr/bin/env python

"""
Test script for the Music Analyzer app
This script tests the basic functionality of each component
"""

import numpy as np
import os
import sys

print("Testing Music Analyzer App components...")

# Test audio input module
print("\nTesting AudioInput module...")
try:
    from app.audio_input import AudioInput
    audio_input = AudioInput()
    print("✓ AudioInput module imported successfully")
except Exception as e:
    print(f"✗ Error importing AudioInput module: {e}")
    
# Test analysis module
print("\nTesting MusicAnalyzer module...")
try:
    from app.analysis import MusicAnalyzer
    analyzer = MusicAnalyzer()
    print("✓ MusicAnalyzer module imported successfully")
    
    # Test with some dummy data
    dummy_data = np.random.rand(22050).astype(np.float32)  # 1 second of random audio
    genre, confidence, components = analyzer.analyze(dummy_data)
    print(f"✓ Analysis produced genre: {genre}, confidence: {confidence:.2f}%")
    print(f"✓ Components analysis contains: {', '.join(components.keys())}")
except Exception as e:
    print(f"✗ Error in MusicAnalyzer module: {e}")
    
# Test visualization module
print("\nTesting visualization module...")
try:
    from app.visualization import (
        create_rhythm_visualization,
        create_melody_visualization,
        create_instrumentation_visualization
    )
    print("✓ Visualization module imported successfully")
    
    # Test with dummy component data
    dummy_rhythm = {"tempo": 120, "tempo_category": "Medium", "complexity": 0.5, "complexity_category": "Moderate"}
    dummy_melody = {"dominant_notes": ["C", "G", "E"], "pitch_variety": 0.15, "variety_category": "Medium", "modality": "Major"}
    dummy_instr = {"brightness": 0.4, "brightness_category": "Balanced", "contrast": 30, "contrast_category": "Balanced", 
                  "timbre_complexity": 20, "complexity_category": "Moderate"}
    
    rhythm_fig = create_rhythm_visualization(dummy_rhythm)
    melody_fig = create_melody_visualization(dummy_melody)
    instr_fig = create_instrumentation_visualization(dummy_instr)
    
    print("✓ All visualization functions worked correctly")
except Exception as e:
    print(f"✗ Error in visualization module: {e}")
    
# Test model module
print("\nTesting model module...")
try:
    from app.models.genre_classifier import GenreClassifier
    model = GenreClassifier(input_features=55, num_genres=10)  # Simplified for testing
    print("✓ GenreClassifier model imported successfully")
    
    # Test feature processing
    dummy_features = {
        "tempo": 120,
        "mfcc": np.random.rand(13, 100),
        "chroma": np.random.rand(12, 100),
        "spectral_centroid": np.random.rand(100),
        "spectral_rolloff": np.random.rand(100),
        "spectral_contrast": np.random.rand(7, 100)
    }
    
    features_tensor = GenreClassifier.preprocess_audio_features(dummy_features)
    print(f"✓ Feature preprocessing worked, tensor shape: {features_tensor.shape}")
    
    # Test prediction
    genres = ["classical", "country", "electronic", "hip-hop", "jazz", 
              "metal", "pop", "reggae", "rock", "blues"]
    genre, confidence = model.predict_genre(features_tensor, genres)
    print(f"✓ Model prediction worked, predicted: {genre}, confidence: {confidence:.2f}%")
except Exception as e:
    print(f"✗ Error in model module: {e}")
    
print("\nTesting complete!")
print("Note: This is a basic test of component imports and functionality.")
print("      A more comprehensive test suite would include unit tests and integration tests.")
print("      For a complete test, run the full application with 'python app.py'.") 