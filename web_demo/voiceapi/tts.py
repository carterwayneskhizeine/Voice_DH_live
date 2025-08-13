from typing import *
import os
import time
import sherpa_onnx
import logging
import numpy as np
import asyncio
import time
import soundfile
from scipy.signal import resample
import io
import re
import threading
import base64
logger = logging.getLogger(__file__)

splitter = re.compile(r'[,，。.!?！？;；、\n]')
_tts_engines = {}

tts_configs = {
    'sherpa-onnx-vits-zh-ll': {
        'model': 'model.onnx',
        'lexicon': 'lexicon.txt',
        'dict_dir': 'dict',
        'tokens': 'tokens.txt',
        'sample_rate': 16000,
        # 'rule_fsts': ['phone.fst', 'date.fst', 'number.fst'],
    },
    'vits-zh-hf-theresa': {
        'model': 'theresa.onnx',
        'lexicon': 'lexicon.txt',
        'dict_dir': 'dict',
        'tokens': 'tokens.txt',
        'sample_rate': 22050,
        # 'rule_fsts': ['phone.fst', 'date.fst', 'number.fst'],
    },
    'vits-melo-tts-zh_en': {
        'model': 'model.onnx',
        'lexicon': 'lexicon.txt',
        'dict_dir': 'dict',
        'tokens': 'tokens.txt',
        'sample_rate': 44100,
        'rule_fsts': ['phone.fst', 'date.fst', 'number.fst'],
    },
}


def load_tts_model(name: str, model_root: str, provider: str, num_threads: int = 1, max_num_sentences: int = 20) -> sherpa_onnx.OfflineTtsConfig:
    cfg = tts_configs[name]
    fsts = []
    model_dir = os.path.join(model_root, name)
    for f in cfg.get('rule_fsts', ''):
        fsts.append(os.path.join(model_dir, f))
    tts_rule_fsts = ','.join(fsts) if fsts else ''

    model_config = sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=os.path.join(model_dir, cfg['model']),
            lexicon=os.path.join(model_dir, cfg['lexicon']),
            dict_dir=os.path.join(model_dir, cfg['dict_dir']),
            tokens=os.path.join(model_dir, cfg['tokens']),
        ),
        provider=provider,
        debug=0,
        num_threads=num_threads,
    )
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=model_config,
        rule_fsts=tts_rule_fsts,
        max_num_sentences=max_num_sentences)

    if not tts_config.validate():
        raise ValueError("tts: invalid config")

    return tts_config


def get_tts_engine(args) -> Tuple[sherpa_onnx.OfflineTts, int]:
    sample_rate = tts_configs[args.tts_model]['sample_rate']
    cache_engine = _tts_engines.get(args.tts_model)
    if cache_engine:
        return cache_engine, sample_rate
    st = time.time()
    tts_config = load_tts_model(
        args.tts_model, args.models_root, args.tts_provider)

    cache_engine = sherpa_onnx.OfflineTts(tts_config)
    elapsed = time.time() - st
    logger.info(f"tts: loaded {args.tts_model} in {elapsed:.2f}s")
    _tts_engines[args.tts_model] = cache_engine

    return cache_engine, sample_rate

# 1. 全局模型管理类
class TTSEngineManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance.engine = None
            return cls._instance

    @classmethod
    def initialize(cls, args):
        instance = cls()
        if instance.engine is None:  # 安全访问属性
            instance.engine, instance.original_sample_rate = get_tts_engine(args)

    @classmethod
    def get_engine(cls):
        instance = cls()  # 确保实例存在
        return instance.engine,instance.original_sample_rate  # 安全访问属性


def clean_text_for_tts(text):
    """清理文本中TTS模型无法处理的字符"""
    if not text or not text.strip():
        return ""
    
    try:
        # 数字转换为中文（在清理其他字符之前处理）
        text = convert_numbers_to_chinese(text)
        
        # 使用字符串替换，避免正则表达式语法错误
        # 移除最常见的问题字符
        chars_to_remove = [
            '×', '＊', '*', '（', '）', '(', ')', 
            '【', '】', '"', '"', '"', '：', ':', '#', 
            '～', '~', ''', ''', '—', '–', '…', '%', '/', '《', '》', '；'
        ]
        
        for char in chars_to_remove:
            text = text.replace(char, '')
        
        # 移除表情符号
        emoji_chars = ['😊', '😄', '🤔', '👍', '💪', '❤️', '🎉', '🔥', '✨']
        for emoji in emoji_chars:
            text = text.replace(emoji, '')
        
        # 移除列表标记
        text = text.replace('- ', '')
        text = text.replace('-', '')
        
        # 多个点号替换为句号
        text = text.replace('...', '。')
        text = text.replace('..', '。')
        
        # 清理多余空白和换行
        while '  ' in text:  # 连续两个空格
            text = text.replace('  ', ' ')
        while '\n\n' in text:  # 连续换行
            text = text.replace('\n\n', '\n')
        text = text.replace('\n', '')  # 移除所有换行
        
        text = text.strip()
        
        # 如果清理后文本为空或过短，返回空
        if len(text) < 2:
            return ""
            
        return text
        
    except Exception as e:
        print(f"文本清理异常: {e}, 原文本: {text}")
        # 最简单的清理
        result = text.replace('*', '').replace('(', '').replace(')', '').replace('#', '').strip()
        return result if len(result) >= 2 else ""

def convert_numbers_to_chinese(text):
    """将文本中的数字转换为中文"""
    try:
        # 手动查找和替换数字
        result = ""
        i = 0
        while i < len(text):
            if text[i].isdigit():
                # 找到数字的开始
                num_start = i
                while i < len(text) and (text[i].isdigit() or text[i] == '.'):
                    i += 1
                num_str = text[num_start:i]
                
                # 转换为中文数字
                try:
                    if '.' in num_str:
                        # 处理小数
                        parts = num_str.split('.')
                        chinese_num = num_to_chinese(int(parts[0])) + "点" + "".join([digit_to_chinese(d) for d in parts[1]])
                    else:
                        chinese_num = num_to_chinese(int(num_str))
                    result += chinese_num
                except:
                    # 如果转换失败，保持原数字
                    result += num_str
            else:
                result += text[i]
                i += 1
        
        return result
    except Exception as e:
        print(f"数字转换异常: {e}")
        return text

def digit_to_chinese(digit):
    """单个数字转中文"""
    digit_map = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四", 
                 "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"}
    return digit_map.get(digit, digit)

def num_to_chinese(num):
    """将数字转换为中文（简单版本）"""
    if num == 0:
        return "零"
    
    digits = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九"]
    units = ["", "十", "百", "千", "万"]
    
    if num < 10:
        return digits[num]
    elif num < 100:
        if num < 20:
            return "一十" + digits[num % 10] if num % 10 != 0 else "十"
        else:
            return digits[num // 10] + "十" + digits[num % 10]
    elif num < 1000:
        return digits[num // 100] + "百" + (num_to_chinese(num % 100) if num % 100 != 0 else "")
    elif num < 10000:
        return digits[num // 1000] + "千" + (num_to_chinese(num % 1000) if num % 1000 != 0 else "")
    else:
        return str(num)  # 对于更大的数字，保持原样

async def get_audio(text, voice_speed=1.0, voice_id=0, target_sample_rate = 16000):
    print("run_tts", text, voice_speed, voice_id)
    
    # 文本预处理
    cleaned_text = clean_text_for_tts(text)
    if not cleaned_text:
        print("文本清理后为空，跳过TTS生成")
        return ""
    
    try:
        # 获取全局共享的TTS引擎
        tts_engine, original_sample_rate = TTSEngineManager.get_engine()

        # 将同步方法放入线程池执行
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None,
            lambda: tts_engine.generate(cleaned_text, voice_id, voice_speed)
        )
        
        # 检查生成的音频是否有效
        if not hasattr(audio, 'samples') or len(audio.samples) == 0:
            print(f"TTS生成的音频为空: {cleaned_text}")
            return ""
            
        samples = audio.samples
        if target_sample_rate != original_sample_rate:
            num_samples = int(
                len(samples) * target_sample_rate / original_sample_rate)
            resampled_chunk = resample(samples, num_samples)
            audio.samples = resampled_chunk.astype(np.float32)
            audio.sample_rate = target_sample_rate

        output = io.BytesIO()
        # 使用 soundfile 写入 WAV 格式数据（自动生成头部）
        soundfile.write(
            output,
            audio.samples,  # 音频数据（numpy 数组）
            samplerate=audio.sample_rate,  # 采样率（如 16000）
            subtype="PCM_16",  # 16-bit PCM 编码
            format="WAV"  # WAV 容器格式
        )

        # 获取字节数据并 Base64 编码
        wav_data = output.getvalue()
        if len(wav_data) == 0:
            print(f"生成的WAV数据为空: {cleaned_text}")
            return ""
            
        return base64.b64encode(wav_data).decode("utf-8")
        
    except Exception as e:
        print(f"TTS生成异常: {e}, 原文本: {text}, 清理后文本: {cleaned_text}")
        return ""

    # import wave
    # import uuid
    # with wave.open('{}.wav'.format(uuid.uuid4()), 'w') as f:
    #     f.setnchannels(1)
    #     f.setsampwidth(2)
    #     f.setframerate(16000)
    #     f.writeframes(samples)
    # return base64.b64encode(samples).decode('utf-8')

