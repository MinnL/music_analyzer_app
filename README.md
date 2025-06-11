# 🎵 Genra - Music Analyzer App

An intelligent, AI-powered web application for analyzing music and audio in real-time. Genra combines traditional machine learning with modern Large Language Models (LLMs) and professional-grade audio analysis tools to provide comprehensive music analysis, genre classification, and educational explanations from microphone input or uploaded audio files.

## ✨ Key Features

- **🎙️ Real-time Audio Analysis**: Capture and analyze audio from your microphone with SDL2-powered audio input
- **📁 Audio File Upload**: Upload music files for analysis (WAV, MP3, OGG, FLAC, M4A)
- **🎧 Audio Playback**: Replay recorded or uploaded audio files with built-in player
- **🎯 High-Confidence Genre Classification**: Advanced CNN-based classifier analyzing multiple audio features across segments
- **🎺 Professional Instrument Detection**: Powered by Essentia algorithms for precise instrument identification
- **⏱️ Extended Audio Analysis**: Processes up to 60 seconds of audio for comprehensive analysis
- **🤖 AI-Powered Explanations**: OpenAI GPT integration for dynamic, contextual music education
- **🔄 Smart Fallback System**: Robust error handling with automatic fallbacks ensuring 100% uptime
- **📊 Interactive Visualizations**: 
  - Rhythm analysis with tempo and beat patterns
  - Melodic analysis with key detection and note distribution
  - Instrumentation analysis with advanced spectral characteristics
- **📱 Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices

## 🎼 Genre Classification

The app uses a sophisticated Convolutional Neural Network (CNN) trained on the GTZAN dataset for music genre classification. The model analyzes mel-spectrograms extracted from audio to identify genres including:

**Supported Genres:**
- 🎵 Blues
- 🎻 Classical  
- 🤠 Country
- 🕺 Disco
- 🎤 Hip Hop
- 🎷 Jazz
- 🤘 Metal
- 🎤 Pop
- 🌴 Reggae
- 🎸 Rock

## 🎺 Advanced Instrument Detection

### Powered by Essentia Professional Audio Analysis

The app features a state-of-the-art instrument detection system powered by **Essentia**, the professional audio analysis library used by Spotify, AcousticBrainz, and other industry leaders.

**Detected Instruments:**
- 🎹 **Piano** - Keyboard/Percussion family
- 🥁 **Drums** - Percussion instruments with rhythmic foundation
- 🎸 **Guitar** - Both acoustic and electric variants
- 🎻 **Violin** - High-pitched bowed string instrument
- 🎻 **Cello** - Large bowed string instrument with rich, warm tone
- 🎺 **Trumpet** - Bright brass instrument
- 🎷 **Saxophone** - Reed woodwind with distinctive character
- 🎸 **Bass** - Low-pitched string instrument providing harmonic foundation

### Advanced Analysis Features

**Essentia-Powered Analysis:**
- **Spectral Feature Extraction**: Professional-grade spectral analysis using SpectralCentroidTime, SpectralContrast, and SpectralComplexity
- **Harmonicity Detection**: Advanced harmonic analysis to distinguish between harmonic and percussive instruments
- **Onset Detection**: Precise attack pattern analysis for instrument classification
- **Robust Fallback**: Automatic degradation to librosa-based detection when needed

**For Each Detected Instrument:**
- **Confidence Score**: Algorithmic confidence in detection accuracy
- **Detailed Description**: Instrument family, characteristics, and musical role
- **Genre Context**: How the instrument fits within the detected genre
- **Technical Characteristics**: Frequency ranges, timbral qualities, and playing techniques

### Technical Implementation

- **Professional Algorithms**: Utilizes Essentia's industry-standard audio analysis algorithms
- **Advanced Classification**: Multi-feature scoring system considering harmonicity, attack patterns, and spectral characteristics
- **Performance Optimized**: Frame-limited processing prevents hanging (max 50 frames per analysis)
- **Error Resilient**: Comprehensive error handling with graceful degradation

## 🚀 Installation

### Prerequisites
- **Python 3.7+** (Python 3.10+ recommended)
- **SDL2** (for audio input on macOS/Linux)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/music_analyzer_app.git
cd music_analyzer_app
```

### Step 2: System Dependencies

**macOS:**
```bash
brew install sdl2 portaudio
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libsdl2-dev portaudio19-dev
```

**Windows:**
```bash
# SDL2 and PortAudio should install automatically via pip
```

### Step 3: Virtual Environment
```bash
python -m venv music_analyzer_env
source music_analyzer_env/bin/activate  # On Windows: music_analyzer_env\Scripts\activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: 🤖 AI Integration Setup (Optional but Recommended)

For enhanced AI-generated explanations:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

📋 **Detailed Setup**: See [LLM_SETUP.md](LLM_SETUP.md) for complete configuration guide.

⚠️ **Note**: App works perfectly without LLM setup using traditional explanations.

## 🎯 Usage

### Starting the Application
```bash
source music_analyzer_env/bin/activate
export OPENAI_API_KEY="your-key-here"  # Optional
python app.py
```

### Using the Web Interface

1. **Open Browser**: Navigate to `http://127.0.0.1:8053/`

2. **Microphone Recording**:
   - Click "Start Recording"
   - Allow microphone access when prompted
   - Record audio (up to 60 seconds for optimal analysis)
   - Click "Stop Recording"

3. **File Upload**:
   - Click "Select an Audio File" or drag & drop
   - Supported: WAV, MP3, OGG, FLAC, M4A (< 10MB)
   - Automatic analysis of first 60 seconds

4. **Audio Playback**:
   - Click "Play Audio" after recording/uploading
   - View source info and duration

5. **Interactive Analysis**:
   - **Visualizations**: Click elements for detailed information
   - **Instrument Details**: View confidence scores and descriptions
   - **AI Explanations**: Get contextual music education (with OpenAI)

## 🔧 Troubleshooting

### Audio Issues
**SDL2 Errors (macOS):**
```bash
brew install sdl2
```

**Microphone Access:**
- Grant browser permissions when prompted
- App automatically switches to file-only mode if mic fails

**PyAudio/PortAudio Issues:**
```bash
# macOS
brew install portaudio
# Linux  
sudo apt-get install portaudio19-dev
```

### Performance Issues
**NumPy Compatibility Warnings:**
- Warnings are harmless and don't affect functionality
- Can be resolved by downgrading: `pip install "numpy<2.0"`

### File Upload Issues
- Ensure supported format (WAV, MP3, OGG, FLAC, M4A)
- Check file size < 10MB
- Clear browser cache if persistent issues

## 🏗️ Technical Architecture

### Core Components

**Audio Processing Pipeline:**
- **SDL2 Audio Input**: Professional-grade audio capture
- **Essentia Analysis Engine**: Industry-standard feature extraction
- **CNN Genre Classifier**: GTZAN-trained deep learning model
- **Advanced Instrument Detector**: Multi-algorithm instrument recognition

**Analysis Modules:**
- **Feature Extraction**: Spectral, timbral, and rhythmic characteristics
- **Segmentation**: Intelligent audio windowing for optimal analysis
- **Classification**: Ensemble methods for robust predictions
- **Visualization**: Interactive Plotly-based charts and graphs

**Web Framework:**
- **Dash/Flask Backend**: Python-based web framework
- **Bootstrap Frontend**: Responsive, mobile-friendly UI
- **Real-time Updates**: WebSocket-based live analysis display

### Advanced Features

**Error Handling & Resilience:**
- **Timeout Protection**: 30-second analysis limits prevent hanging
- **Graceful Degradation**: Multiple fallback systems ensure functionality
- **Smart Recovery**: Automatic error handling with user-friendly messages

**Performance Optimizations:**
- **Caching System**: Analysis results cached for repeated access
- **Frame Limiting**: Processing caps prevent resource exhaustion
- **Parallel Processing**: Multi-threaded analysis where possible

## 🆕 Latest Updates (v3.0 - Professional Audio Integration)

### Major Improvements

**🎯 Essentia Integration:**
- Professional-grade audio analysis using Spotify's choice library
- Advanced spectral feature extraction (SpectralCentroidTime, SpectralContrast, SpectralComplexity)
- Industry-standard harmonicity and onset detection algorithms

**🎺 Enhanced Instrument Detection:**
- Dramatically improved accuracy using professional algorithms
- Specific instrument identification instead of generic "Mixed Instruments"
- Confidence scoring and detailed instrument characteristics

**🔧 Robust Error Handling:**
- Comprehensive timeout protection preventing app hanging
- Multi-level fallback systems (Essentia → librosa → basic analysis)
- Smart error recovery with user-friendly messaging

**🎨 UI/UX Improvements:**
- SDL2-powered audio input for professional-grade recording
- Enhanced visualizations with detailed instrument information
- Improved mobile responsiveness and accessibility

**⚡ Performance Optimizations:**
- Frame-limited processing prevents resource exhaustion
- Intelligent caching system for repeated analysis
- Optimized audio segmentation for faster processing

**🤖 AI Integration Enhancements:**
- Fixed LLM explanation formatting and display issues
- Web-optimized AI responses with structured sections
- Enhanced prompt engineering for better music education

### Technical Achievements

- **Zero-Downtime Operation**: Robust fallback systems ensure 100% uptime
- **Professional Audio Quality**: SDL2 and Essentia integration
- **Industry-Standard Analysis**: Same tools used by Spotify and major platforms
- **Intelligent Error Recovery**: Self-healing system architecture
- **Cross-Platform Compatibility**: Tested on macOS, Linux, and Windows

## 📚 Development

### Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run with debug mode
DEBUG=1 python app.py
```

## 📋 Requirements

**System Requirements:**
- Python 3.7+ (3.10+ recommended)
- 4GB RAM minimum (8GB recommended)
- Modern web browser with HTML5 support
- Microphone (optional - file upload available)

**Key Dependencies:**
- `essentia-tensorflow>=2.1b6` - Professional audio analysis
- `librosa>=0.8.0` - Audio feature extraction
- `torch>=1.9.0` - Deep learning framework
- `dash>=2.0.0` - Web framework
- `plotly>=5.0.0` - Interactive visualizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Essentia Team** - Professional audio analysis toolkit
- **Librosa Developers** - Audio and music analysis library
- **Dash/Plotly Team** - Interactive visualization framework
- **GTZAN Dataset** - Music genre classification dataset
- **OpenAI** - GPT integration for intelligent explanations
- **Contributors** - All developers who helped improve this project

---

**📞 Support**: Open an issue for bug reports or feature requests  
**🌟 Star**: If you find this project useful, please star it on GitHub  
**🔄 Updates**: Watch this repository for the latest features and improvements 