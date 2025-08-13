import os
from openai import OpenAI

# 从环境变量获取LLM配置，支持多种提供商
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "doubao")  # 默认使用豆包
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")

# 预设配置
PROVIDER_CONFIGS = {
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model_name": "doubao-seed-1.6-250615"
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat"
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-3.5-turbo"
    }
}

# 确定最终配置
config = PROVIDER_CONFIGS.get(LLM_PROVIDER, {})
base_url = LLM_BASE_URL or config.get("base_url", "")
model_name = LLM_MODEL_NAME or config.get("model_name", "")
api_key = LLM_API_KEY

# 验证配置
if not api_key:
    raise ValueError(f"您必须设置环境变量 LLM_API_KEY。当前提供商: {LLM_PROVIDER}")

if not base_url:
    raise ValueError(f"无法确定 {LLM_PROVIDER} 的 base_url，请设置 LLM_BASE_URL 环境变量")

if not model_name:
    raise ValueError(f"无法确定 {LLM_PROVIDER} 的 model_name，请设置 LLM_MODEL_NAME 环境变量")

print(f"LLM配置: 提供商={LLM_PROVIDER}, 模型={model_name}")

llm_client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)


def llm_stream(prompt):
    stream = llm_client.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
        model=model_name,
        messages=[
            {"role": "system", "content": "你是人工智能助手"},
            {"role": "user", "content": prompt},
        ],
        # 响应内容是否流式返回
        stream=True,
    )
    return stream
