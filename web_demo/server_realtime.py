import json
import os
from contextlib import asynccontextmanager
import re
import asyncio
import base64
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, UploadFile, File,HTTPException,WebSocketDisconnect,WebSocket
from voiceapi.asr import start_asr_stream, ASRResult,ASREngineManager
import uvicorn
import argparse
from voiceapi.llm import llm_stream
from voiceapi.tts import get_audio,TTSEngineManager

# 2. 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服务启动时初始化模型（示例参数）
    print("ASR模型正在初始化，请稍等")
    ASREngineManager.initialize(samplerate=16000, args = args)
    print("TTS模型正在初始化，请稍等")
    TTSEngineManager.initialize(args = args)
    yield
    # 服务关闭时清理资源
    engine = ASREngineManager.get_engine()
    if engine and hasattr(engine, 'cleanup'):
        engine.cleanup()
    print("服务已关闭")


app = FastAPI(lifespan=lifespan)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="web_demo/static"), name="static")


def split_sentence(sentence, min_length=10):
    # 定义包括小括号在内的主要标点符号
    punctuations = r'[。？！；…，、()（）]'
    # 使用正则表达式切分句子，保留标点符号
    parts = re.split(f'({punctuations})', sentence)
    parts = [p for p in parts if p]  # 移除空字符串
    sentences = []
    current = ''
    for part in parts:
        if current:
            # 如果当前片段加上新片段长度超过最小长度，则将当前片段添加到结果中
            if len(current) + len(part) >= min_length:
                sentences.append(current + part)
                current = ''
            else:
                current += part
        else:
            current = part
    # 将剩余的片段添加到结果中
    if len(current) >= 2:
        sentences.append(current)
    return sentences

PUNCTUATION_SET = {
    '，', " ", '。', '！', '？', '；', '：', '、', '（', '）', '【', '】', '“', '”',
    ',', '.', '!', '?', ';', ':', '(', ')', '[', ']', '"', "'"
}

async def gen_stream(prompt, asr = False, voice_speed=None, voice_id=None):
    print("gen_stream", voice_speed, voice_id)
    if asr:
        chunk = {
            "prompt": prompt
        }
        yield f"{json.dumps(chunk)}\n"  # 使用换行符分隔 JSON 块

    # Streaming:
    print("----- streaming request -----")
    stream = llm_stream(prompt)
    llm_answer_cache = ""
    for chunk in stream:
        if not chunk.choices:
            continue
        llm_answer_cache += chunk.choices[0].delta.content

        # 查找第一个标点符号的位置
        punctuation_pos = -1
        for i, char in enumerate(llm_answer_cache[8:]):
            if char in PUNCTUATION_SET:
                punctuation_pos = i + 8
                break
        # 如果找到标点符号且第一小句字数大于8
        if punctuation_pos != -1:
            # 获取第一小句
            first_sentence = llm_answer_cache[:punctuation_pos + 1]
            # 剩余的文字
            remaining_text = llm_answer_cache[punctuation_pos + 1:]
            print("get_audio: ", first_sentence)
            
            # 尝试生成音频，如果失败则跳过这一句
            try:
                base64_string = await get_audio(first_sentence, voice_id=voice_id, voice_speed=voice_speed)
                if base64_string:  # 只有在成功生成音频时才发送
                    chunk = {
                        "text": first_sentence,
                        "audio": base64_string,
                        "endpoint": False
                    }
                    # 更新缓存为剩余的文字
                    llm_answer_cache = remaining_text
                    yield f"{json.dumps(chunk)}\n"  # 使用换行符分隔 JSON 块
                    await asyncio.sleep(0.2)  # 模拟异步延迟
                else:
                    # 音频生成失败，只更新缓存，跳过这一句
                    print(f"音频生成失败，跳过文本: {first_sentence}")
                    llm_answer_cache = remaining_text
            except Exception as e:
                print(f"TTS生成异常，跳过文本: {first_sentence}, 错误: {e}")
                # 发生异常时，只更新缓存，跳过这一句
                llm_answer_cache = remaining_text
    print("get_audio: ", llm_answer_cache)
    base64_string = ""
    if len(llm_answer_cache) >= 2:
        try:
            base64_string = await get_audio(llm_answer_cache, voice_id=voice_id, voice_speed=voice_speed)
            if not base64_string:
                print(f"最后一句音频生成失败，跳过文本: {llm_answer_cache}")
        except Exception as e:
            print(f"最后一句TTS生成异常，跳过文本: {llm_answer_cache}, 错误: {e}")
            base64_string = ""
    
    chunk = {
            "text": llm_answer_cache if base64_string else "",  # 如果音频生成失败，不返回文本
            "audio": base64_string,
            "endpoint": True
    }
    yield f"{json.dumps(chunk)}\n"  # 使用换行符分隔 JSON 块

@app.websocket("/asr")
async def websocket_asr(websocket: WebSocket, samplerate: int = 16000):
    await websocket.accept()

    asr_stream = await start_asr_stream(samplerate, args)
    if not asr_stream:
        print("failed to start ASR stream")
        await websocket.close()
        return

    async def task_recv_pcm():
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive(), timeout=1.0)
                # print(f"message: {data}")
            except asyncio.TimeoutError:
                continue  # 没有数据到达，继续循环

            if "text" in data.keys():
                print(f"Received text message: {data}")
                data = data["text"]
                if data.strip() == "vad":
                    print("VAD signal received")
                    await asr_stream.vad_touched()
            elif "bytes" in data.keys():
                pcm_bytes = data["bytes"]
                print("XXXX pcm_bytes", len(pcm_bytes))
                if not pcm_bytes:
                    return
                await asr_stream.write(pcm_bytes)


    async def task_send_result():
        while True:
            result: ASRResult = await asr_stream.read()
            if not result:
                return
            await websocket.send_json(result.to_dict())
    try:
        await asyncio.gather(task_recv_pcm(), task_send_result())
    except WebSocketDisconnect:
        print("asr: disconnected")
    finally:
        await asr_stream.close()

@app.post("/eb_stream")    # 前端调用的path
async def eb_stream(request: Request):
    try:
        body = await request.json()
        input_mode = body.get("input_mode")
        voice_speed = body.get("voice_speed", 1.0)
        voice_id = body.get("voice_id", 0)

        if voice_speed == "":
            voice_speed = 1.0
        if voice_id == "":
            voice_id = 0

        if input_mode == "text":
            prompt = body.get("prompt")
            return StreamingResponse(gen_stream(prompt, asr=False, voice_speed=voice_speed, voice_id=voice_id), media_type="application/json")
        else:
            raise HTTPException(status_code=400, detail="Invalid input mode")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 启动Uvicorn服务器
if __name__ == "__main__":
    models_root = './models'

    for d in ['.', '..', 'web_demo']:
        if os.path.isdir(f'{d}/models'):
            models_root = f'{d}/models'
            break

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888, help="port number")
    parser.add_argument("--addr", type=str,
                        default="0.0.0.0", help="serve address")

    parser.add_argument("--asr-provider", type=str,
                        default="cpu", help="asr provider, cpu or cuda")
    parser.add_argument("--tts-provider", type=str,
                        default="cpu", help="tts provider, cpu or cuda")

    parser.add_argument("--threads", type=int, default=2,
                        help="number of threads")

    parser.add_argument("--models-root", type=str, default=models_root,
                        help="model root directory")

    parser.add_argument("--asr-model", type=str, default='zipformer-bilingual',
                        help="ASR model name: zipformer-bilingual, sensevoice, paraformer-trilingual, paraformer-en, whisper-medium")

    parser.add_argument("--asr-lang", type=str, default='zh',
                        help="ASR language, zh, en, ja, ko, yue")

    parser.add_argument("--tts-model", type=str, default='sherpa-onnx-vits-zh-ll',
                        help="TTS model name: vits-zh-hf-theresa, vits-melo-tts-zh_en")

    args = parser.parse_args()

    if args.tts_model == 'vits-melo-tts-zh_en' and args.tts_provider == 'cuda':
        print(
            "vits-melo-tts-zh_en does not support CUDA fallback to CPU")
        args.tts_provider = 'cpu'

    uvicorn.run(app, host=args.addr, port=args.port)
