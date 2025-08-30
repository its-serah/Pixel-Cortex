#!/usr/bin/env python3
"""
Standalone Integration Test for Performance Monitoring and Audio Processing

This test validates that the core audio processing packages are installed 
and basic functionality works without requiring the full FastAPI application.
"""

import pytest
import sys
import os
import time


def test_audio_processing_imports():
    """Test that all audio processing packages can be imported successfully"""
    
    # Test OpenAI Whisper
    try:
        import whisper
        print("✅ OpenAI Whisper imported successfully")
    except ImportError as e:
        pytest.fail(f"❌ Failed to import whisper: {e}")
    
    # Test librosa
    try:
        import librosa
        print("✅ librosa imported successfully")
    except ImportError as e:
        pytest.fail(f"❌ Failed to import librosa: {e}")
    
    # Test soundfile
    try:
        import soundfile as sf
        print("✅ soundfile imported successfully")
    except ImportError as e:
        pytest.fail(f"❌ Failed to import soundfile: {e}")
    
    # Test SpeechRecognition
    try:
        import speech_recognition as sr
        print("✅ SpeechRecognition imported successfully")
    except ImportError as e:
        pytest.fail(f"❌ Failed to import speech_recognition: {e}")
    
    # Test pydub
    try:
        import pydub
        print("✅ pydub imported successfully")
    except ImportError as e:
        pytest.fail(f"❌ Failed to import pydub: {e}")
    
    # Test webrtcvad
    try:
        import webrtcvad
        print("✅ webrtcvad imported successfully")
    except ImportError as e:
        pytest.fail(f"❌ Failed to import webrtcvad: {e}")


def test_basic_audio_functionality():
    """Test basic functionality of audio processing libraries"""
    
    try:
        import numpy as np
        import librosa
        
        # Create a simple test audio signal (1 second of 440Hz sine wave)
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Test librosa functionality
        # Load the test signal into librosa's expected format
        y = test_audio.astype(np.float32)
        
        # Test basic audio processing
        mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13)
        assert mfccs.shape[0] == 13, "MFCC should have 13 coefficients"
        
        # Test spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sample_rate)
        assert len(spectral_centroids) > 0, "Should compute spectral centroids"
        
        print("✅ Basic librosa audio processing works correctly")
        
    except Exception as e:
        pytest.fail(f"❌ Basic audio functionality test failed: {e}")


def test_speech_recognition_setup():
    """Test SpeechRecognition microphone setup"""
    
    try:
        import speech_recognition as sr
        
        # Test microphone recognition setup (without actually recording)
        recognizer = sr.Recognizer()
        
        # Test different audio sources are available
        mic_list = sr.Microphone.list_microphone_names()
        print(f"✅ Found {len(mic_list)} microphone sources")
        
        # Test recognizer adjustment for ambient noise (simulation)
        with sr.Microphone() as source:
            # This will work if a microphone is available
            recognizer.adjust_for_ambient_noise(source, duration=0.1)
            print("✅ SpeechRecognition microphone setup successful")
        
    except sr.RequestError as e:
        print(f"⚠️ SpeechRecognition API error (expected in CI): {e}")
        # This is expected in CI environments without microphones
        pass
    except Exception as e:
        print(f"⚠️ SpeechRecognition setup issue (may be expected): {e}")
        # Don't fail the test for microphone-related issues in CI
        pass


def test_whisper_model_loading():
    """Test that Whisper models can be loaded"""
    
    try:
        import whisper
        
        # Load the smallest Whisper model for testing
        print("Loading Whisper tiny model...")
        model = whisper.load_model("tiny")
        assert model is not None, "Whisper model should load successfully"
        print("✅ Whisper tiny model loaded successfully")
        
        # Test basic model info
        assert hasattr(model, 'transcribe'), "Model should have transcribe method"
        print("✅ Whisper model has transcribe method")
        
    except Exception as e:
        pytest.fail(f"❌ Whisper model loading failed: {e}")


def test_performance_monitoring_basics():
    """Test basic performance monitoring without full app"""
    
    try:
        import psutil
        import time
        
        # Test CPU usage monitoring
        cpu_percent = psutil.cpu_percent(interval=0.1)
        assert 0 <= cpu_percent <= 100, f"CPU percentage should be 0-100, got {cpu_percent}"
        
        # Test memory monitoring
        memory = psutil.virtual_memory()
        assert memory.total > 0, "Total memory should be positive"
        assert 0 <= memory.percent <= 100, "Memory percentage should be 0-100"
        
        # Test disk monitoring
        disk = psutil.disk_usage('/')
        assert disk.total > 0, "Disk total should be positive"
        
        print("✅ Basic performance monitoring works")
        print(f"   CPU: {cpu_percent}%")
        print(f"   Memory: {memory.percent}% ({memory.used / 1e9:.1f}GB / {memory.total / 1e9:.1f}GB)")
        print(f"   Disk: {(disk.used / disk.total) * 100:.1f}%")
        
    except Exception as e:
        pytest.fail(f"❌ Performance monitoring test failed: {e}")


def test_ml_and_nlp_imports():
    """Test that ML/NLP packages required for the app can be imported"""
    
    try:
        # Test scikit-learn
        import sklearn
        print("✅ scikit-learn imported successfully")
        
        # Test NLTK
        import nltk
        print("✅ NLTK imported successfully")
        
        # Test spaCy
        import spacy
        print("✅ spaCy imported successfully")
        
        # Test spaCy model loading
        nlp = spacy.load("en_core_web_sm")
        assert nlp is not None, "spaCy English model should load"
        print("✅ spaCy English model loaded successfully")
        
        # Test basic spaCy functionality
        doc = nlp("VPN connection requires MFA authentication.")
        assert len(doc) > 0, "spaCy should parse text"
        tokens = [token.text for token in doc]
        assert "VPN" in tokens, "Should tokenize VPN"
        print("✅ spaCy basic functionality works")
        
    except Exception as e:
        pytest.fail(f"❌ ML/NLP imports failed: {e}")


def test_database_and_api_imports():
    """Test database and API-related imports"""
    
    try:
        # Test SQLAlchemy
        import sqlalchemy
        print("✅ SQLAlchemy imported successfully")
        
        # Test FastAPI
        import fastapi
        print("✅ FastAPI imported successfully")
        
        # Test pydantic with settings
        from pydantic_settings import BaseSettings
        print("✅ pydantic-settings imported successfully")
        
        # Test pytest
        import pytest
        print("✅ pytest imported successfully")
        
        # Test redis (may not be connected but should import)
        import redis
        print("✅ redis imported successfully")
        
    except Exception as e:
        pytest.fail(f"❌ Database/API imports failed: {e}")


if __name__ == "__main__":
    print("🧪 Running Pixel Cortex Standalone Integration Tests")
    print("=" * 60)
    
    # Run all tests
    tests = [
        test_audio_processing_imports,
        test_basic_audio_functionality,
        test_speech_recognition_setup,
        test_whisper_model_loading,
        test_performance_monitoring_basics,
        test_ml_and_nlp_imports,
        test_database_and_api_imports,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            print(f"\n🔍 Running {test_func.__name__}...")
            test_func()
            print(f"✅ {test_func.__name__} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All integration tests passed!")
        sys.exit(0)
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        sys.exit(1)
