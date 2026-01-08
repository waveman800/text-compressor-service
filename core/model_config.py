import os
from dataclasses import dataclass
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMProviderConfig:
    provider_name: str
    api_base_url: str
    api_key: str
    api_version: Optional[str] = None


@dataclass
class ModelConfig:
    name: str
    context_length: int
    max_output_tokens: int
    description: str = ""
    tokenizer: Optional[str] = None
    provider: Optional[str] = None


def get_llm_providers() -> Dict[str, LLMProviderConfig]:
    providers = {}
    
    if os.getenv("OPENAI_API_KEY"):
        providers["openai"] = LLMProviderConfig(
            provider_name="OpenAI",
            api_base_url=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    if os.getenv("AZURE_OPENAI_API_KEY"):
        providers["azure"] = LLMProviderConfig(
            provider_name="Azure OpenAI",
            api_base_url=os.getenv("AZURE_OPENAI_API_BASE_URL"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )
    
    if os.getenv("ANTHROPIC_API_KEY"):
        providers["anthropic"] = LLMProviderConfig(
            provider_name="Anthropic",
            api_base_url=os.getenv("ANTHROPIC_API_BASE_URL", "https://api.anthropic.com"),
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
    
    if os.getenv("VLLM_API_BASE_URL"):
        providers["vllm"] = LLMProviderConfig(
            provider_name="vLLM",
            api_base_url=os.getenv("VLLM_API_BASE_URL"),
            api_key=os.getenv("VLLM_API_KEY", "")
        )
    
    if os.getenv("OLLAMA_API_BASE_URL"):
        providers["ollama"] = LLMProviderConfig(
            provider_name="Ollama",
            api_base_url=os.getenv("OLLAMA_API_BASE_URL"),
            api_key=os.getenv("OLLAMA_API_KEY", "")
        )
    
    if os.getenv("LOCALAI_API_BASE_URL"):
        providers["localai"] = LLMProviderConfig(
            provider_name="LocalAI",
            api_base_url=os.getenv("LOCALAI_API_BASE_URL"),
            api_key=os.getenv("LOCALAI_API_KEY", "")
        )
    
    if os.getenv("QWEN_API_KEY"):
        providers["qwen"] = LLMProviderConfig(
            provider_name="Qwen",
            api_base_url=os.getenv("QWEN_API_BASE_URL"),
            api_key=os.getenv("QWEN_API_KEY")
        )
    
    if os.getenv("BAICHUAN_API_KEY"):
        providers["baichuan"] = LLMProviderConfig(
            provider_name="Baichuan",
            api_base_url=os.getenv("BAICHUAN_API_BASE_URL"),
            api_key=os.getenv("BAICHUAN_API_KEY")
        )
    
    if os.getenv("ZHIPU_API_KEY"):
        providers["zhipu"] = LLMProviderConfig(
            provider_name="Zhipu AI",
            api_base_url=os.getenv("ZHIPU_API_BASE_URL"),
            api_key=os.getenv("ZHIPU_API_KEY")
        )
    
    if os.getenv("DEEPSEEK_API_KEY"):
        providers["deepseek"] = LLMProviderConfig(
            provider_name="DeepSeek",
            api_base_url=os.getenv("DEEPSEEK_API_BASE_URL"),
            api_key=os.getenv("DEEPSEEK_API_KEY")
        )
    
    if os.getenv("YI_API_KEY"):
        providers["yi"] = LLMProviderConfig(
            provider_name="Yi",
            api_base_url=os.getenv("YI_API_BASE_URL"),
            api_key=os.getenv("YI_API_KEY")
        )
    
    if os.getenv("MOONSHOT_API_KEY"):
        providers["moonshot"] = LLMProviderConfig(
            provider_name="Moonshot",
            api_base_url=os.getenv("MOONSHOT_API_BASE_URL"),
            api_key=os.getenv("MOONSHOT_API_KEY")
        )
    
    if os.getenv("CUSTOM_QWEN_API_KEY"):
        providers["custom_qwen"] = LLMProviderConfig(
            provider_name="Custom Qwen Service",
            api_base_url=os.getenv("CUSTOM_QWEN_API_BASE_URL"),
            api_key=os.getenv("CUSTOM_QWEN_API_KEY")
        )
    
    return providers


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "gpt-3.5-turbo": ModelConfig(
        name="GPT-3.5 Turbo",
        context_length=4096,
        max_output_tokens=4096,
        description="OpenAI的GPT-3.5 Turbo模型",
        provider="openai"
    ),
    "gpt-4": ModelConfig(
        name="GPT-4",
        context_length=8192,
        max_output_tokens=4096,
        description="OpenAI的GPT-4模型",
        provider="openai"
    ),
    "gpt-4-turbo": ModelConfig(
        name="GPT-4 Turbo",
        context_length=128000,
        max_output_tokens=4096,
        description="OpenAI的GPT-4 Turbo模型",
        provider="openai"
    ),
    "gpt-4-turbo-preview": ModelConfig(
        name="GPT-4 Turbo Preview",
        context_length=128000,
        max_output_tokens=4096,
        description="OpenAI的GPT-4 Turbo Preview模型",
        provider="openai"
    ),
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        context_length=128000,
        max_output_tokens=4096,
        description="OpenAI的GPT-4o模型",
        provider="openai"
    ),
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o Mini",
        context_length=128000,
        max_output_tokens=16384,
        description="OpenAI的GPT-4o Mini模型",
        provider="openai"
    ),
    "claude-3-opus": ModelConfig(
        name="Claude 3 Opus",
        context_length=200000,
        max_output_tokens=4096,
        description="Anthropic的Claude 3 Opus模型",
        provider="anthropic"
    ),
    "claude-3-sonnet": ModelConfig(
        name="Claude 3 Sonnet",
        context_length=200000,
        max_output_tokens=4096,
        description="Anthropic的Claude 3 Sonnet模型",
        provider="anthropic"
    ),
    "claude-3-haiku": ModelConfig(
        name="Claude 3 Haiku",
        context_length=200000,
        max_output_tokens=4096,
        description="Anthropic的Claude 3 Haiku模型",
        provider="anthropic"
    ),
    "claude-3.5-sonnet": ModelConfig(
        name="Claude 3.5 Sonnet",
        context_length=200000,
        max_output_tokens=8192,
        description="Anthropic的Claude 3.5 Sonnet模型",
        provider="anthropic"
    ),
    "llama-2-7b": ModelConfig(
        name="LLaMA 2 7B",
        context_length=4096,
        max_output_tokens=2048,
        description="Meta的LLaMA 2 7B模型",
        provider="vllm"
    ),
    "llama-2-70b": ModelConfig(
        name="LLaMA 2 70B",
        context_length=4096,
        max_output_tokens=2048,
        description="Meta的LLaMA 2 70B模型",
        provider="vllm"
    ),
    "llama-3-8b": ModelConfig(
        name="LLaMA 3 8B",
        context_length=8192,
        max_output_tokens=4096,
        description="Meta的LLaMA 3 8B模型",
        provider="vllm"
    ),
    "llama-3-70b": ModelConfig(
        name="LLaMA 3 70B",
        context_length=8192,
        max_output_tokens=4096,
        description="Meta的LLaMA 3 70B模型",
        provider="vllm"
    ),
    "llama-3.1-8b": ModelConfig(
        name="LLaMA 3.1 8B",
        context_length=128000,
        max_output_tokens=4096,
        description="Meta的LLaMA 3.1 8B模型",
        provider="vllm"
    ),
    "llama-3.1-70b": ModelConfig(
        name="LLaMA 3.1 70B",
        context_length=128000,
        max_output_tokens=4096,
        description="Meta的LLaMA 3.1 70B模型",
        provider="vllm"
    ),
    "mistral-7b": ModelConfig(
        name="Mistral 7B",
        context_length=8192,
        max_output_tokens=2048,
        description="Mistral AI的7B模型",
        provider="vllm"
    ),
    "mixtral-8x7b": ModelConfig(
        name="Mixtral 8x7B",
        context_length=32768,
        max_output_tokens=4096,
        description="Mistral AI的8x7B模型",
        provider="vllm"
    ),
    "qwen3-14b-awq": ModelConfig(
        name="Qwen3 14B AWQ",
        context_length=120000,
        max_output_tokens=8192,
        description="局域网内部署的Qwen3 14B AWQ模型",
        provider="custom_qwen"
    ),
    "qwen-72b-chat": ModelConfig(
        name="Qwen 72B Chat",
        context_length=32768,
        max_output_tokens=2048,
        description="阿里云的Qwen 72B Chat模型",
        provider="qwen"
    ),
    "qwen-110b": ModelConfig(
        name="Qwen 110B",
        context_length=32768,
        max_output_tokens=8192,
        description="阿里云的Qwen 110B模型",
        provider="qwen"
    ),
    "baichuan-13b": ModelConfig(
        name="Baichuan 13B",
        context_length=4096,
        max_output_tokens=2048,
        description="百度的Baichuan 13B模型",
        provider="baichuan"
    ),
    "baichuan2-13b": ModelConfig(
        name="Baichuan2 13B",
        context_length=4096,
        max_output_tokens=2048,
        description="百度的Baichuan2 13B模型",
        provider="baichuan"
    ),
    "deepseek-chat": ModelConfig(
        name="DeepSeek Chat",
        context_length=32768,
        max_output_tokens=4096,
        description="DeepSeek的Chat模型",
        provider="deepseek"
    ),
    "deepseek-coder": ModelConfig(
        name="DeepSeek Coder",
        context_length=16384,
        max_output_tokens=4096,
        description="DeepSeek的Coder模型",
        provider="deepseek"
    ),
    "DeepSeek-V3": ModelConfig(
        name="DeepSeek V3",
        context_length=120000,
        max_output_tokens=32000,
        description="DeepSeek的V3模型",
        provider="custom_qwen"
    ),
    "yi-34b": ModelConfig(
        name="Yi 34B",
        context_length=200000,
        max_output_tokens=4096,
        description="零一万物Yi 34B模型",
        provider="yi"
    ),
    "yi-34b-chat": ModelConfig(
        name="Yi 34B Chat",
        context_length=200000,
        max_output_tokens=4096,
        description="零一万物Yi 34B Chat模型",
        provider="yi"
    ),
    "internlm2-20b": ModelConfig(
        name="InternLM2 20B",
        context_length=32768,
        max_output_tokens=4096,
        description="书生浦语InternLM2 20B模型",
        provider="zhipu"
    ),
    "chatglm3-6b": ModelConfig(
        name="ChatGLM3 6B",
        context_length=8192,
        max_output_tokens=2048,
        description="智谱ChatGLM3 6B模型",
        provider="zhipu"
    ),
    "chatglm3-6b-32k": ModelConfig(
        name="ChatGLM3 6B 32K",
        context_length=32768,
        max_output_tokens=2048,
        description="智谱ChatGLM3 6B 32K模型",
        provider="zhipu"
    ),
    "Qwen3-14B-AWQ": ModelConfig(
        name="Qwen3 14B AWQ",
        context_length=120000,
        max_output_tokens=64000,
        description="局域网内部署的Qwen3 14B AWQ模型",
        provider="custom_qwen"
    ),
}


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    if model_name not in MODEL_CONFIGS:
        return None
    return MODEL_CONFIGS[model_name]


def get_all_models() -> Dict[str, ModelConfig]:
    return MODEL_CONFIGS


def add_custom_model(model_name: str, config: ModelConfig):
    MODEL_CONFIGS[model_name] = config


def calculate_session_len(model_name: str, safety_margin: float = 0.1) -> Optional[int]:
    config = get_model_config(model_name)
    if config is None:
        return None
    return int(config.context_length * (1 - safety_margin))


def get_provider_for_model(model_name: str) -> Optional[LLMProviderConfig]:
    config = get_model_config(model_name)
    if config is None or config.provider is None:
        return None
    
    providers = get_llm_providers()
    return providers.get(config.provider)


def get_available_models() -> Dict[str, ModelConfig]:
    providers = get_llm_providers()
    available_models = {}
    
    for model_id, config in MODEL_CONFIGS.items():
        if config.provider is None or config.provider in providers:
            available_models[model_id] = config
    
    return available_models
