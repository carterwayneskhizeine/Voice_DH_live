import os
from openai import OpenAI
import contextlib

# ===== 代理管理辅助函数 =====
def disable_proxy_temporarily():
    """临时禁用代理环境变量，返回上下文管理器"""
    @contextlib.contextmanager
    def proxy_context():
        old_proxies = {}
        proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
        
        # 保存并删除代理环境变量
        for var in proxy_vars:
            if var in os.environ:
                old_proxies[var] = os.environ[var]
                del os.environ[var]
        
        try:
            yield
        finally:
            # 恢复代理设置
            for var, value in old_proxies.items():
                os.environ[var] = value
    
    return proxy_context()

# ===== 配置参数（直接写在代码中，可修改切换） =====
LLM_PROVIDER = "coze"  # 可选: doubao, deepseek, openai, coze

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
    },
    "coze": {
        "base_url": "https://api.coze.cn",
        "bot_id": "7538267516649537545",
        "user_id": "123456789"
    }
}

# ===== 敏感信息从环境变量获取 =====
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
COZE_API_TOKEN = os.getenv("COZE_API_TOKEN", "")

# ===== 初始化配置 =====
config = PROVIDER_CONFIGS.get(LLM_PROVIDER, {})

if LLM_PROVIDER == "coze":
    # COZE配置验证
    if not COZE_API_TOKEN:
        raise ValueError("使用COZE提供商时必须设置环境变量 COZE_API_TOKEN")
    
    # 导入COZE相关依赖
    try:
        from cozepy import COZE_CN_BASE_URL, Coze, TokenAuth, Message, ChatEventType
        
        # 临时禁用代理环境变量来初始化COZE客户端
        with disable_proxy_temporarily():
            # 初始化COZE客户端
            coze_client = Coze(
                auth=TokenAuth(token=COZE_API_TOKEN), 
                base_url=COZE_CN_BASE_URL
            )
            bot_id = config["bot_id"]
            user_id = config["user_id"]
            
            print(f"LLM配置: 提供商=COZE, Bot ID={bot_id}")
        
    except ImportError:
        raise ValueError("使用COZE需要安装cozepy: pip install cozepy")
        
else:
    # OpenAI格式配置验证
    if not LLM_API_KEY:
        raise ValueError(f"您必须设置环境变量 LLM_API_KEY。当前提供商: {LLM_PROVIDER}")
    
    base_url = config.get("base_url", "")
    model_name = config.get("model_name", "")
    
    if not base_url or not model_name:
        raise ValueError(f"未找到提供商 {LLM_PROVIDER} 的配置")
    
    print(f"LLM配置: 提供商={LLM_PROVIDER}, 模型={model_name}")
    
    llm_client = OpenAI(
        base_url=base_url,
        api_key=LLM_API_KEY,
    )


# ===== COZE流式响应适配器 =====
class CozeStreamAdapter:
    """将COZE流式响应适配为OpenAI格式"""
    def __init__(self, coze_stream):
        self.coze_stream = coze_stream
    
    def __iter__(self):
        for event in self.coze_stream:
            if event.event == ChatEventType.CONVERSATION_MESSAGE_DELTA:
                # 转换为OpenAI格式的chunk
                yield self._create_openai_chunk(event.message.content)
    
    def _create_openai_chunk(self, content):
        """模拟OpenAI的响应格式"""
        return type('ChatCompletionChunk', (), {
            'choices': [type('Choice', (), {
                'delta': type('Delta', (), {'content': content})()
            })()]
        })()


# ===== 统一的流式接口 =====
def llm_stream(prompt):
    """统一的LLM流式接口，根据配置的提供商自动切换"""
    if LLM_PROVIDER == "coze":
        return coze_stream(prompt)
    else:
        return openai_stream(prompt)


def coze_stream(prompt):
    """COZE流式响应实现"""
    # 临时禁用代理环境变量，避免与Clash冲突
    with disable_proxy_temporarily():
        coze_stream = coze_client.chat.stream(
            bot_id=bot_id,
            user_id=user_id,
            additional_messages=[
                Message.build_user_question_text(prompt)
            ],
        )
        return CozeStreamAdapter(coze_stream)


def openai_stream(prompt):
    """OpenAI格式流式响应实现"""
    stream = llm_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是人工智能助手"},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    return stream
