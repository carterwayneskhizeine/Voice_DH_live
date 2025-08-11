# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Always respond in Simplified Chinese

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda create -n dh_live python=3.11
conda activate dh_live

# Install PyTorch (GPU version)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install PyTorch (CPU version, if no GPU)
pip install torch

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

**Gradio Interface (Recommended for first-time users):**
```bash
python app.py
```

**Video Data Preparation:**
```bash
# Process video for mini version
python data_preparation_mini.py video_data/000002/video.mp4 video_data/000002

# Prepare web assets
python data_preparation_web.py video_data/000002
```

**Offline Video Generation (Windows only):**
```bash
# Generate video with audio file (16kHz mono WAV required)
python demo_mini.py video_data/000002/assets video_data/audio0.wav output.mp4
```

**Web Demo Servers:**
```bash
# Simple web demo
python web_demo/server.py
# Access: http://localhost:8888/static/MiniLive.html

# Real-time voice conversation (requires ASR/TTS models)
python web_demo/server_realtime.py
# Access: http://localhost:8888/static/MiniLive_RealTime.html
```

### Model Download
Download checkpoint files from:
- [BaiduDrive](https://pan.baidu.com/s/1jH3WrIAfwI3U5awtnt9KPQ?pwd=ynd7)
- [GoogleDrive](https://drive.google.com/drive/folders/1az5WEWOFmh0_yrF3I9DEyctMyjPolo8V?usp=sharing)

Place downloaded models in the `checkpoint/` directory.

## Architecture Overview

### Core Components

**DH_Live_mini System:**
- Ultra-lightweight 2D talking head solution (39 MFlops per frame)
- Works on any device without GPU requirements
- Web-based real-time inference capability

**Key Modules:**
- `talkingface/` - Core talking head models and utilities
  - `models/DINet_mini.py` - Lightweight neural network for face animation
  - `models/audio2bs_lstm.py` - Audio to blendshape conversion
  - `render_model_mini.py` - Mini rendering pipeline
- `mini_live/` - OpenGL rendering and 3D face mesh handling
- `web_demo/` - Web server and real-time conversation system
- `data_preparation_*.py` - Video preprocessing pipelines

**Real-time Conversation Pipeline:**
1. VAD (Voice Activity Detection) - Web client
2. ASR (Automatic Speech Recognition) - Server local
3. LLM (Large Language Model) - Cloud service
4. TTS (Text-to-Speech) - Server local  
5. Digital Human Rendering - Web client

### Configuration

**LLM Integration:** Edit `web_demo/voiceapi/llm.py` to configure your LLM API:
```python
# Example for DeepSeek
base_url = "https://api.deepseek.com"
api_key = "your-api-key-here"
model_name = "deepseek-chat"
```

**Asset Management:**
- Place character assets in `web_demo/static/assets/` or `web_demo/static/assets2/`
- Video processing creates `combined_data.json.gz` for web rendering
- Face textures and 3D meshes stored in respective asset folders

### Platform Support

| Platform | Video Processing | Offline Generation | Web Server | Real-time Chat |
|----------|-----------------|-------------------|------------|----------------|
| Windows  | ✅              | ✅                | ✅         | ✅             |
| Linux/macOS | ✅           | ❌                | ✅         | ✅             |

### Important Notes

- WebCodecs API requires HTTPS or localhost for web demo
- Audio files must be 16kHz mono WAV format
- The system uses minimal computational resources (works on 2-core 4GB systems)
- Commercial deployment requires logo removal authorization from matesx.com

### Dependencies

Core dependencies include:
- PyTorch (with optional CUDA support)
- MediaPipe (face detection and landmarks)
- OpenGL libraries (PyOpenGL, glfw, pyglm)
- Audio processing (sherpa-onnx for ASR/TTS)
- Web framework (FastAPI, Gradio)