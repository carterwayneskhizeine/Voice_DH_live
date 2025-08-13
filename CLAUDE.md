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

### Model Download and Setup
Download checkpoint files from:
- [BaiduDrive](https://pan.baidu.com/s/1jH3WrIAfwI3U5awtnt9KPQ?pwd=ynd7)
- [GoogleDrive](https://drive.google.com/drive/folders/1az5WEWOFmh0_yrF3I9DEyctMyjPolo8V?usp=sharing)

Place downloaded models in the `checkpoint/` directory.

For real-time conversation, download additional models to `models/`:
- ASR: sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
- TTS: sherpa-onnx-vits-zh-ll

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

## Architecture Overview

### Core System Architecture

**DH_Live_mini System:**
- Ultra-lightweight 2D talking head solution (39 MFlops per frame)
- Works on any device without GPU requirements
- Web-based real-time inference using WebCodecs API and WebAssembly

**Data Flow:**
```
Raw Video → data_preparation_mini.py → Processed Data (.pkl)
                ↓
         data_preparation_web.py → Web Assets (combined_data.json.gz)
                ↓
           Web Client (WASM) → Real-time Rendering
```

### Key Modules

**Core Inference Pipeline:**
- `talkingface/models/DINet_mini.py` - Lightweight face animation network
- `talkingface/models/audio2bs_lstm.py` - Audio-to-blendshape LSTM model
- `talkingface/render_model_mini.py` - PyTorch-based mini rendering pipeline
- `mini_live/render.py` - OpenGL rendering with texture fusion
- `mini_live/opengl_render_interface.py` - OpenGL wrapper for 3D face mesh

**Data Processing:**
- `talkingface/mediapipe_utils.py` - MediaPipe face detection and landmarks
- `mini_live/obj/` - 3D face mesh utilities and OBJ file processing
- `data_preparation_*.py` - Video preprocessing and asset generation pipelines

**Web Interface:**
- `web_demo/static/DHLiveMini.wasm` - WebAssembly inference engine
- `web_demo/static/js/DHLiveMini.js` - JavaScript client-side inference wrapper
- `web_demo/server_realtime.py` - FastAPI server with WebSocket support
- `web_demo/voiceapi/` - ASR/LLM/TTS integration modules

### Real-time Conversation Pipeline

**Component Distribution:**
| Component | Location | Technology |
|-----------|----------|------------|
| VAD (Voice Activity Detection) | Web Client | JavaScript |
| ASR (Speech Recognition) | Server Local | sherpa-onnx |
| LLM (Large Language Model) | Cloud Service | OpenAI API Compatible |
| TTS (Text-to-Speech) | Server Local | sherpa-onnx |
| Digital Human Rendering | Web Client | WebAssembly + WebGL |

**Data Flow:**
```
Audio Input (Web) → WebSocket → ASR (Server) → LLM (Cloud) → TTS (Server) 
                                                    ↓
Web Rendering ← WebSocket ← Audio Stream + Text Response
```

### Configuration

**LLM Integration:** Edit `web_demo/voiceapi/llm.py`:
```python
# DeepSeek example
base_url = "https://api.deepseek.com"
api_key = "your-api-key-here"
model_name = "deepseek-chat"

# Doubao example  
base_url = "https://ark.cn-beijing.volces.com/api/v3"
api_key = "your-api-key-here"
model_name = "doubao-pro-32k-character-241215"
```

**Asset Management:**
- Character assets: `web_demo/static/assets/` and `web_demo/static/assets2/`
- Video processing generates `combined_data.json.gz` for web rendering
- Face textures and 3D meshes stored in respective asset folders
- Replace assets to change digital human appearance

### Platform Support

| Platform | Video Processing | Offline Generation | Web Server | Real-time Chat |
|----------|-----------------|-------------------|------------|----------------|
| Windows  | ✅              | ✅                | ✅         | ✅             |
| Linux/macOS | ✅           | ❌                | ✅         | ✅             |

### Technical Requirements

- **Browser Support:** WebCodecs API requires HTTPS or localhost
- **Audio Format:** 16kHz mono WAV files for offline processing
- **Hardware:** Works on 2-core 4GB systems (ultra-lightweight design)
- **Commercial Use:** Logo removal requires authorization from matesx.com

### Dependencies

**Core Inference:**
- PyTorch (with optional CUDA support)
- MediaPipe (face detection and landmarks)
- OpenGL libraries (PyOpenGL, glfw, pyglm)

**Web Services:**
- FastAPI/Gradio (web framework)
- sherpa-onnx (ASR/TTS)
- WebAssembly runtime (browser-based inference)

**Data Processing:**
- scikit-learn, kaldi_native_fbank (audio processing)
- tqdm (progress bars)
- Various audio/video codecs via system ffmpeg