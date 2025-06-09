# 🎵 Genra - Music Analyzer App

An intelligent, AI-powered web application for analyzing music and audio in real-time. Genra combines traditional machine learning with modern Large Language Models (LLMs) to provide comprehensive music analysis, genre classification, and educational explanations from microphone input or uploaded audio files.

## Features

- **Real-time Audio Analysis**: Capture and analyze audio from your microphone
- **Audio File Upload**: Upload music files for analysis (WAV, MP3, OGG, FLAC, M4A)
- **Audio Playback**: Replay recorded or uploaded audio files
- **High-Confidence Genre Classification**: Utilizes an advanced classifier analyzing multiple audio features across segments for improved accuracy and confidence.
- **Enhanced Instrument Detection**: Accurately identifies musical instruments such as piano, drums, guitar, and more with advanced spectral analysis
- **Extended Audio Analysis**: Processes up to 60 seconds of audio for more comprehensive analysis.
- **🤖 AI-Powered Explanations**: Integrated OpenAI GPT models to generate dynamic, contextual, and educational explanations for genre classifications with proper web-optimized formatting
- **🔄 Smart Fallback System**: Automatically falls back to traditional explanations when LLM is unavailable, ensuring reliability
- **Interactive Visualizations**: 
  - Rhythm analysis with tempo and beat patterns
  - Melodic analysis with key detection and note distribution
  - Instrumentation analysis with timbral characteristics
- **Responsive Design**: Works on desktop and tablet devices

## Pre-trained Model

The app uses a Convolutional Neural Network (CNN) trained on the GTZAN dataset for music genre classification. The model analyzes mel-spectrograms extracted from audio to identify genres including:

- Blues
- Classical
- Country
- Disco
- Hip Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## Instrument Detection

The app features an advanced instrument detection system that can identify various musical instruments in audio recordings, including:

- Piano
- Drums
- Bass
- Acoustic and Electric Guitar
- Violin
- Cello
- Trumpet
- Saxophone
- Synthesizer

The instrument detector uses a sophisticated algorithm that analyzes:
- Spectral characteristics of each instrument
- Attack and decay patterns
- Harmonic content analysis
- Onset detection for percussion instruments
- Harmonic-percussive separation

For each detected instrument, the app provides:
- Confidence score
- Description of the instrument
- Role of the instrument in the detected music genre

## Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/music_analyzer_app.git
   cd music_analyzer_app
   ```

2. Create and activate a virtual environment
   ```
   python -m venv music_analyzer_env
   source music_analyzer_env/bin/activate  # On Windows: music_analyzer_env\Scripts\activate
   ```

3. Install the dependencies
   ```
   pip install -r requirements.txt
   ```

   Note: You may need to install PortAudio first for PyAudio to work properly.
   - On macOS: `brew install portaudio`
   - On Linux: `sudo apt-get install portaudio19-dev`
   - On Windows: PyAudio should install directly through pip

4. **🤖 Optional: Set up AI-Powered Explanations (Recommended)**
   
   For enhanced AI-generated explanations using OpenAI GPT:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```
   
   📋 **Detailed Setup Instructions**: See [LLM_SETUP.md](LLM_SETUP.md) for complete setup guide including:
   - OpenAI API key acquisition
   - Environment configuration 
   - Troubleshooting tips
   
   ⚠️ **Note**: The app works perfectly without LLM setup - it automatically falls back to traditional explanations.

## Usage

1. Start the application
   ```
   python app.py
   ```

2. Open your web browser and navigate to: http://127.0.0.1:8053/ (Note: Port might differ, check terminal output)

3. For microphone recording:
   - Click "Start Recording"
   - Allow microphone access when prompted
   - Speak or play music (up to 60 seconds recommended for best analysis)
   - Click "Stop Recording" when finished

4. For file upload:
   - Click "Select an Audio File" or drag and drop a file
   - Supported formats: WAV, MP3, OGG, FLAC, M4A
   - The app will automatically analyze the first 60 seconds of the file

5. To replay audio:
   - After recording or uploading, click "Play Audio"
   - The player will show source information and duration

6. Interact with visualizations:
   - Click on elements to see detailed information
   - Hover over graphs for additional data points

7. Instrument detection:
   - The app automatically detects instruments in the analyzed audio
   - View detected instruments, their confidence scores, and roles in the "Instrument Detection" section

Note: The application now analyzes up to 60 seconds of audio by default to provide a more thorough genre classification.

## Troubleshooting

### Microphone Issues
- On macOS, you may see PortAudio errors. This is a known issue with PyAudio on Mac systems.
- The app will automatically switch to demo mode if microphone access fails.
- You can still use the file upload feature to analyze audio.

### File Upload Issues
- Ensure your audio file is in one of the supported formats (WAV, MP3, OGG, FLAC, M4A)
- Files should be less than 10MB for optimal performance

### Playback Issues
- If playback doesn't work, ensure your browser supports HTML5 audio

## Requirements

- Python 3.7 or higher
- Web browser with HTML5 and JavaScript support
- Microphone (optional, since file upload is supported)

## Technical Architecture

- **Audio Input Module**: Captures audio from microphone or processes uploaded files (up to 60 seconds).
- **Analysis Module**: Processes audio data and extracts features for classification using the HighConfidenceClassifier.
- **High-Confidence Genre Classifier**: Employs a custom classifier that analyzes various audio characteristics (tempo, spectral features, harmony, etc.) across multiple 20-second segments of the audio sample (up to 60 seconds total). It uses genre-specific fingerprints and ensemble methods to achieve higher confidence and more robust predictions.
- **Instrument Detection System**: Uses advanced spectral analysis, onset detection, and harmonic-percussive separation to identify musical instruments with high accuracy.
- **Visualization Module**: Creates interactive visual representations of audio analysis.
- **Web Interface**: Dash/Flask-based responsive UI with Bootstrap styling.

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 🆕 Latest Updates (v2.0 - AI Integration)

- **🤖 LLM Integration**: Complete integration with OpenAI GPT-4o-mini for intelligent music analysis explanations
- **🎨 Web-Optimized Explanations**: AI-generated explanations formatted specifically for web display with structured sections:
  - 🎼 What Makes [Genre] Special?
  - 🔍 Analysis Results  
  - 💡 Cool Discovery
  - 🎵 Musical Context
- **🔧 Fixed HTML Rendering**: Resolved React component serialization issues for proper AI explanation display
- **✅ Robust Error Handling**: Enhanced error handling and graceful fallback to traditional explanations
- **🔄 Smart Environment Management**: Automatic detection and configuration of virtual environments
- **📊 Enhanced Debug Logging**: Comprehensive logging for LLM interactions and system diagnostics
- **🎯 Improved Prompts**: Web-focused prompts that generate concise, educational, and engaging explanations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The librosa team for their audio and music analysis tools
- The Dash and Plotly teams for their data visualization libraries 