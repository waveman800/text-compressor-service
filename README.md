# Text Compression Service

动态文本压缩服务，基于lmdeploy 0.8.0的动态压缩逻辑，支持OpenAI协议兼容的代理服务。

## 功能特性

- **动态文本压缩**：基于文本重要性和上下文窗口约束，动态压缩文本内容
- **聊天历史压缩**：智能压缩聊天历史，保留重要信息
- **OpenAI协议兼容**：支持OpenAI协议的代理服务，可与各种类OpenAI协议的模型服务集成
- **灵活配置**：可配置会话窗口大小、压缩模式等参数
- **分词器支持**：支持Hugging Face的AutoTokenizer，字符级别的回退机制

## 项目结构

```
text_compressor_service/
├── core/                    # 核心压缩逻辑
│   ├── dynamic_compressor.py   # 动态文本压缩器
│   └── tokenizer_wrapper.py    # 分词器包装器
├── api/                     # API服务
│   └── main.py                 # FastAPI应用
├── examples/                # 使用示例
│   ├── http_client_example.py  # HTTP客户端示例
│   └── direct_client_example.py # 直接调用示例
├── tests/                   # 测试用例
│   └── test_core.py            # 核心功能测试
├── requirements.txt         # 依赖列表
├── setup.py                 # 安装配置
└── README.md                # 项目说明
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
cd api
python3 main.py
```

服务将在 `http://0.0.0.0:8000` 启动。

### API文档

访问 `http://localhost:8000/docs` 查看Swagger UI文档。

## API端点

### 文本压缩

```
POST /compress/text
```

参数：
- `text`: 要压缩的文本
- `current_prompt`: 当前的提示词（可选）
- `max_new_tokens`: 生成新token的最大数量（可选，默认256）
- `session_len`: 会话窗口的最大token数（可选，默认4096）
- `use_fast_compression`: 是否使用快速压缩模式（可选，默认False）

### 聊天历史压缩

```
POST /compress/chat
```

参数：
- `chat_history`: 聊天历史列表
- `current_prompt`: 当前的提示词（可选）
- `max_new_tokens`: 生成新token的最大数量（可选，默认256）
- `session_len`: 会话窗口的最大token数（可选，默认4096）

### OpenAI协议兼容端点

```
POST /v1/chat/completions
```

符合OpenAI API规范的聊天完成端点，会自动压缩聊天历史。

```
GET /v1/models
```

获取可用模型列表。

## 配置

可以通过环境变量配置目标OpenAI服务：

- `OPENAI_API_BASE_URL`: OpenAI API基础URL（默认：http://localhost:8000/v1）
- `OPENAI_API_KEY`: OpenAI API密钥

## 使用示例

### 直接调用核心API

```python
from core.dynamic_compressor import DynamicContextCompressor
from core.tokenizer_wrapper import TokenizerWrapper

# 初始化
compressor = DynamicContextCompressor(session_len=4096)

# 压缩文本
long_text = "这里是一段很长的文本..."
compressed_text, was_compressed = compressor.dynamic_compress(long_text)

# 压缩聊天历史
chat_history = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！我是AI助手。"},
    # ... 更多消息
]
compressed_chat, was_compressed = compressor.compress_chat_history(chat_history)
```

### 使用HTTP客户端

```python
import requests

# 压缩文本
response = requests.post(
    "http://localhost:8000/compress/text",
    json={
        "text": "这里是一段很长的文本...",
        "session_len": 4096
    }
)
print(response.json())

# 调用OpenAI兼容API
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！我是AI助手。"}
        ]
    }
)
print(response.json())
```

## 测试

```bash
python3 -m pytest tests/
```

## 开发

### 安装开发依赖

```bash
pip install -e .[dev]
```

### 代码格式化

```bash
black .
isort .
```

## 许可证

MIT License
