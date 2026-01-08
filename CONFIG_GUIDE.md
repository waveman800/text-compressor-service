# 文本压缩服务配置指南

## 概述

文本压缩服务支持通过环境变量配置多个外接LLM服务提供商，实现模型感知的动态文本压缩。服务会根据请求的模型自动选择对应的服务提供商，并根据模型的上下文窗口限制动态调整压缩参数。

## 配置文件

在项目根目录的 `.env` 文件中配置您的LLM服务。

### 支持的服务提供商

#### 1. OpenAI

```bash
OPENAI_API_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-your-openai-api-key
```

#### 2. Azure OpenAI

```bash
AZURE_OPENAI_API_BASE_URL=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

#### 3. Anthropic Claude

```bash
ANTHROPIC_API_BASE_URL=https://api.anthropic.com
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key
```

#### 4. vLLM (本地部署)

```bash
VLLM_API_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=
```

#### 5. Ollama (本地部署)

```bash
OLLAMA_API_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=
```

#### 6. LocalAI (本地部署)

```bash
LOCALAI_API_BASE_URL=http://localhost:8080/v1
LOCALAI_API_KEY=
```

#### 7. 通义千问 (Qwen)

```bash
QWEN_API_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=sk-your-qwen-api-key
```

#### 8. 百川 (Baichuan)

```bash
BAICHUAN_API_BASE_URL=https://api.baichuan-ai.com/v1
BAICHUAN_API_KEY=sk-your-baichuan-api-key
```

#### 9. 智谱AI (Zhipu)

```bash
ZHIPU_API_BASE_URL=https://open.bigmodel.cn/api/paas/v4
ZHIPU_API_KEY=your-zhipu-api-key
```

#### 10. DeepSeek

```bash
DEEPSEEK_API_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_API_KEY=sk-your-deepseek-api-key
```

#### 11. 零一万物 (Yi)

```bash
YI_API_BASE_URL=https://api.lingyiwanwu.com/v1
YI_API_KEY=sk-your-yi-api-key
```

#### 12. 月之暗面 (Moonshot)

```bash
MOONSHOT_API_BASE_URL=https://api.moonshot.cn/v1
MOONSHOT_API_KEY=sk-your-moonshot-api-key
```

### 服务配置

```bash
# 压缩服务端口
SERVICE_PORT=8000

# 默认模型（未指定时使用）
DEFAULT_MODEL=gpt-3.5-turbo

# 安全余量（0-1），预留上下文窗口的比例
SAFETY_MARGIN=0.1

# 调试模式
DEBUG=false
```

## 使用示例

### 1. 启动服务

```bash
cd /home/ai/dev/text_compressor_service
python -m api.main
```

### 2. 查看可用的模型

```bash
curl http://localhost:8000/models
```

响应示例：
```json
{
  "models": [
    {
      "id": "gpt-3.5-turbo",
      "name": "GPT-3.5 Turbo",
      "context_length": 4096,
      "max_output_tokens": 4096,
      "description": "OpenAI的GPT-3.5 Turbo模型",
      "provider": "openai"
    }
  ]
}
```

### 3. 查看配置的服务提供商

```bash
curl http://localhost:8000/providers
```

响应示例：
```json
{
  "providers": [
    {
      "id": "openai",
      "name": "OpenAI",
      "api_base_url": "https://api.openai.com/v1",
      "has_api_key": true
    }
  ]
}
```

### 4. 使用OpenAI协议发送请求

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "请压缩这段文本：人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"}
    ],
    "max_tokens": 256
  }'
```

### 5. 使用Python requests库

```python
import requests

url = "http://localhost:8000/v1/chat/completions"

payload = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {
            "role": "user",
            "content": "请压缩这段文本：人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
        }
    ],
    "temperature": 0.7
}

response = requests.post(url, json=payload)
print(response.json())
```

### 6. 使用OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {
            "role": "user",
            "content": "请压缩这段文本：人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
        }
    ]
)

print(response.choices[0].message.content)
```

## 支持的模型

### OpenAI模型
- gpt-3.5-turbo
- gpt-4
- gpt-4-turbo
- gpt-4-turbo-preview
- gpt-4o
- gpt-4o-mini

### Anthropic模型
- claude-3-opus
- claude-3-sonnet
- claude-3-haiku
- claude-3.5-sonnet

### Meta模型 (vLLM)
- llama-2-7b
- llama-2-70b
- llama-3-8b
- llama-3-70b
- llama-3.1-8b
- llama-3.1-70b
- mistral-7b
- mixtral-8x7b

### 阿里云模型
- qwen3-14b-awq
- qwen-72b-chat
- qwen-110b

### 百川模型
- baichuan-13b
- baichuan2-13b

### DeepSeek模型
- deepseek-chat
- deepseek-coder

### 零一万物模型
- yi-34b
- yi-34b-chat

### 智谱模型
- internlm2-20b
- chatglm3-6b
- chatglm3-6b-32k

## 工作原理

1. **模型选择**：客户端在请求中指定 `model` 参数
2. **提供商匹配**：服务根据模型名称查找对应的服务提供商
3. **动态压缩**：根据模型的上下文窗口限制自动计算压缩阈值
4. **请求代理**：将压缩后的请求转发到对应的服务提供商

## 注意事项

1. 确保在 `.env` 文件中正确配置了至少一个服务提供商的API密钥
2. 不同的服务提供商可能需要不同的API格式，本服务已针对OpenAI兼容接口进行了适配
3. 安全余量（SAFETY_MARGIN）建议设置为0.1-0.2之间，以确保不会超出模型的上下文限制
4. 本地部署的服务（如vLLM、Ollama）需要确保服务正在运行且可访问

## 故障排除

### 问题：请求失败，提示"No provider configured for model"

**解决方案**：检查 `.env` 文件中是否配置了对应服务提供商的API密钥

### 问题：压缩效果不佳

**解决方案**：
- 调整 `SAFETY_MARGIN` 参数
- 尝试使用不同的模型
- 检查输入文本的长度和复杂度

### 问题：连接超时

**解决方案**：
- 确保外接服务正在运行
- 检查网络连接
- 验证API基础URL配置是否正确
