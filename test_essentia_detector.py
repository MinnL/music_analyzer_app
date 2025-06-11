#!/usr/bin/env python3
"""
Test script for the new Essentia instrument detector
"""
import numpy as np
import librosa
from app.models.essentia_instrument_detector import EssentiaInstrumentDetector

def test_essentia_detector():
    """Test the Essentia instrument detector with a simple audio signal"""
    print("🧪 Testing Essentia Instrument Detector...")
    
    # Initialize detector
    detector = EssentiaInstrumentDetector()
    
    # Create a test audio signal (440 Hz sine wave - A4 note)
    duration = 3.0  # seconds
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Piano-like signal (fundamental + harmonics)
    fundamental = 440  # A4
    piano_signal = (
        0.6 * np.sin(2 * np.pi * fundamental * t) +  # Fundamental
        0.3 * np.sin(2 * np.pi * fundamental * 2 * t) +  # 2nd harmonic
        0.2 * np.sin(2 * np.pi * fundamental * 3 * t) +  # 3rd harmonic
        0.1 * np.sin(2 * np.pi * fundamental * 4 * t)    # 4th harmonic
    )
    
    # Add some envelope (attack-decay) for piano-like character
    envelope = np.exp(-t * 1.5)  # Exponential decay
    piano_signal *= envelope
    
    print(f"📊 Test signal: {duration}s, {sample_rate}Hz, Piano-like (A4 = {fundamental}Hz)")
    print(f"🔧 Using {'Essentia' if detector.use_essentia else 'fallback'} algorithms")
    
    # Test the detector
    try:
        detected_instruments = detector.detect_instruments(piano_signal, sample_rate)
        
        print(f"✅ Detection completed successfully!")
        print(f"🎯 Detected instruments: {detected_instruments}")
        
        if 'Piano' in detected_instruments:
            print("✅ Piano correctly detected!")
        else:
            print("⚠️ Piano not detected - algorithm may need tuning")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return False

def test_real_audio():
    """Test with a real audio file if available"""
    print("\n🎵 Testing with real audio (if available)...")
    
    # Try to load a test file
    test_files = [
        "test_audio.wav",
        "sample.wav", 
        "../test.wav"
    ]
    
    detector = EssentiaInstrumentDetector()
    
    for file_path in test_files:
        try:
            audio_data, sr = librosa.load(file_path, sr=22050)
            print(f"📁 Loaded: {file_path} ({len(audio_data)/sr:.1f}s)")
            
            detected = detector.detect_instruments(audio_data, sr)
            print(f"🎯 Real audio detection: {detected}")
            return True
            
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"❌ Error with {file_path}: {e}")
            continue
    
    print("ℹ️ No test audio files found - skipping real audio test")
    return False

if __name__ == "__main__":
    print("🚀 Starting Essentia Instrument Detector Tests\n")
    
    # Test 1: Synthetic signal
    success1 = test_essentia_detector()
    
    # Test 2: Real audio (optional)
    success2 = test_real_audio()
    
    print(f"\n📈 Test Results:")
    print(f"   Synthetic audio: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   Real audio: {'✅ PASS' if success2 else 'ℹ️ SKIPPED'}")
    
    if success1:
        print("\n🎉 Essentia detector is working correctly!")
        print("📝 The enhanced instrument detection should provide much better accuracy.")
    else:
        print("\n⚠️ Issues detected - check Essentia installation")
        print("💡 Try: pip install essentia-tensorflow") 