# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在处理此代码库时提供指导。

始终使用简体中文回复

## 开发命令

### 环境配置
```bash
# 创建conda环境
conda create -n dh_live python=3.11
conda activate dh_live

# 安装PyTorch (GPU版本)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 安装PyTorch (CPU版本，如果没有GPU)
pip install torch

# 安装依赖
pip install -r requirements.txt

# 安装COZE支持 (可选)
pip install cozepy
```

### 模型下载和设置
从以下链接下载检查点文件：
- [百度网盘](https://pan.baidu.com/s/1jH3WrIAfwI3U5awtnt9KPQ?pwd=ynd7)
- [GoogleDrive](https://drive.google.com/drive/folders/1az5WEWOFmh0_yrF3I9DEyctMyjPolo8V?usp=sharing)

将下载的模型放在 `checkpoint/` 目录中。

用于实时对话，需下载额外模型到 `models/`：
- ASR: sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
- TTS: **vits-melo-tts-zh_en** (推荐，支持中英文) 或 sherpa-onnx-vits-zh-ll (仅中文)

**下载中英文TTS模型：**
```bash
cd models
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2
tar -xjf vits-melo-tts-zh_en.tar.bz2
```

### 运行应用程序

**Gradio界面 (首次用户推荐):**
```bash
python app.py
```

**视频数据准备:**
```bash
# 处理mini版本视频
python data_preparation_mini.py video_data/000002/video.mp4 video_data/000002

# 准备网页资源
python data_preparation_web.py video_data/000002
```

**离线视频生成 (仅Windows):**
```bash
# 使用音频文件生成视频 (需要16kHz单声道WAV)
python demo_mini.py video_data/000002/assets video_data/audio0.wav output.mp4
```

**网页演示服务器:**
```bash
# 简单网页演示
python web_demo/server.py
# 访问: http://localhost:8888/static/MiniLive.html

# 实时语音对话 (需要ASR/TTS模型)
python web_demo/server_realtime.py
# 访问: http://localhost:8888/static/MiniLive_RealTime.html
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

### 配置说明

**LLM集成配置:** 编辑 `web_demo/voiceapi/llm.py`：

**支持的提供商：**
- `doubao` - 豆包 (默认)
- `deepseek` - DeepSeek
- `openai` - OpenAI
- `coze` - COZE对话机器人

**切换提供商：**
```python
# 修改第29行的LLM_PROVIDER值
LLM_PROVIDER = "doubao"  # 可选: doubao, deepseek, openai, coze
```

**环境变量配置：**
```bash
# 豆包配置
export LLM_API_KEY="your-doubao-api-key"

# COZE配置 (需要先设置LLM_PROVIDER="coze")
export COZE_API_TOKEN="pat_ge5xxxx"
```

**COZE Bot配置 (在代码中)：**
```python
# 在PROVIDER_CONFIGS["coze"]中修改
"bot_id": "7538267516649537545",  # 你的Bot ID
"user_id": "123456789"           # 用户ID
```

**代理冲突解决：**
- 系统已自动处理Clash代理与COZE API的冲突
- COZE调用时会临时禁用代理，调用完成后自动恢复
- 无需手动关闭代理服务

**资源管理:**
- 人物资源: `web_demo/static/assets/` 和 `web_demo/static/assets2/`
- 视频处理生成 `combined_data.json.gz` 用于网页渲染
- 面部纹理和3D网格存储在相应资源文件夹
- 替换资源以更改数字人外观

### 平台支持

| 平台 | 视频处理 | 离线生成 | 网页服务器 | 实时对话 |
|------|---------|---------|-----------|---------|
| Windows  | ✅ | ✅ | ✅ | ✅ |
| Linux/macOS | ✅ | ❌ | ✅ | ✅ |

### 技术要求

- **浏览器支持:** WebCodecs API 需要 HTTPS 或 localhost
- **音频格式:** 离线处理需要16kHz单声道WAV文件
- **硬件要求:** 在2核4GB系统上运行 (超轻量级设计)
- **商业使用:** Logo移除需要获得 matesx.com 授权

### 重要修改记录

**🔧 TTS模型改进 (2025.08.15)**
- 默认TTS模型已更改为 `vits-melo-tts-zh_en` (支持中英文)
- 修复了原模型无法处理英文、数字、特殊符号的问题
- 添加了智能文本清理功能，自动处理OOV字符
- 数字自动转换为中文 (如: 2023 → 二千零二十三)

**🔧 COZE对话机器人集成 (2025.08.15)**
- 新增COZE提供商支持，可替代豆包等传统LLM
- 统一的流式响应接口，无缝切换不同提供商
- 自动处理代理冲突，支持Clash等代理工具
- 环境变量安全管理敏感信息

**🔧 代理冲突解决方案**
- 程序级别自动绕过代理，解决socks5h协议冲突
- 临时禁用代理环境变量，API调用完成后自动恢复
- 无需手动关闭Clash等代理服务

**🔧 异常处理改进**
- TTS生成失败时自动跳过，继续处理后续文本
- 完整的错误日志记录，便于问题诊断
- 流式响应中断保护，避免整个对话流程中断

### 故障排除

**ASR语音识别不工作:**
1. 确保访问 `http://localhost:8888/static/MiniLive_RealTime.html`
2. 点击语音输入区域并允许麦克风权限
3. 检查浏览器控制台是否有WebSocket连接错误
4. 确认ASR模型文件存在于 `models/` 目录

**TTS语音合成问题:**
- 英文不播报: 确保使用 `vits-melo-tts-zh_en` 模型
- 数字不播报: 系统已自动转换为中文
- 特殊符号错误: 已实现智能文本清理

**COZE API调用失败:**
- 检查 `COZE_API_TOKEN` 环境变量是否设置
- 代理冲突: 系统已自动处理，无需手动操作
- Bot ID配置: 确认 `bot_id` 在代码中正确设置

### 依赖项

**核心推理:**
- PyTorch (可选CUDA支持)
- MediaPipe (人脸检测和关键点)
- OpenGL库 (PyOpenGL, glfw, pyglm)

**网页服务:**
- FastAPI/Gradio (网页框架)
- sherpa-onnx (ASR/TTS)
- WebAssembly运行时 (浏览器端推理)
- cozepy (COZE API支持)

**数据处理:**
- scikit-learn, kaldi_native_fbank (音频处理)
- tqdm (进度条)
- 各种音频/视频编解码器 (通过系统ffmpeg)