# 超长文本压缩与大模型交互使用指南

## 快速开始

### 方法1: 使用LongTextProcessor（推荐）

```python
from examples.long_text_processing_example import LongTextProcessor

# 1. 初始化处理器
processor = LongTextProcessor(
    model_name="gpt-3.5-turbo",
    context_length=4096,      # 模型上下文窗口
    max_output_tokens=512,    # 最大输出token数
    enable_compression=True,  # 启用压缩
    safety_margin=0.1         # 预留10%安全余量
)

# 2. 处理文本并查询
result = processor.process_and_query(
    text=long_text,           # 超长文本
    prompt="请总结这段文字",   # 提示词
    model_api_func=your_api_func  # 可选：大模型API函数
)

# 3. 获取结果
if result['compression_info']:
    print(f"压缩比: {result['compression_info']['compression_ratio']:.2%}")
print(f"模型响应: {result['model_response']}")
```

### 方法2: 使用LLMClientWithCompression（集成真实API）

```python
from examples.real_llm_integration_example import LLMClientWithCompression

# 1. 初始化客户端
client = LLMClientWithCompression(
    api_type="openai",                    # API类型
    api_key="your-openai-api-key",        # API密钥
    model_name="gpt-3.5-turbo",
    context_length=4096,
    max_output_tokens=512,
    enable_compression=True
)

# 2. 调用（自动处理压缩）
result = client.chat_with_long_context(
    messages=[{"role": "user", "content": long_text}],
    system_prompt="你是一个专业的AI助手。",
    temperature=0.7
)

# 3. 获取结果
print(result['response']['content'])
```

## 完整流程说明

### 步骤1: 检查文本长度
```
文本tokens: 10890
提示词tokens: 28
预留输出tokens: 1024
总计tokens: 11942
可用上下文: 3686
⚠️  需要压缩！超出 8256 tokens
```

### 步骤2: 执行文本压缩
```
✅ 压缩完成
  原始tokens: 10890
  压缩后tokens: 2702
  压缩比: 24.81%
  压缩耗时: 0.016秒
```

### 步骤3: 构建大模型请求
```
最终输入tokens: 2730
上下文使用率: 74.07%
请求已构建，准备发送给模型
```

### 步骤4: 调用大模型API
```
✅ 模型调用成功
```

## 支持的API类型

### 1. OpenAI API
```python
client = LLMClientWithCompression(
    api_type="openai",
    api_key="sk-...",
    model_name="gpt-3.5-turbo",
    context_length=4096
)
```

### 2. Anthropic Claude API
```python
client = LLMClientWithCompression(
    api_type="anthropic",
    api_key="sk-ant-...",
    model_name="claude-3-sonnet-20240229",
    context_length=200000  # Claude支持200K上下文
)
```

### 3. 自定义API（本地模型）
```python
client = LLMClientWithCompression(
    api_type="custom",
    api_key="dummy-key",
    base_url="http://localhost:8000/v1",
    model_name="local-model",
    context_length=4096
)
```

## 使用场景

### 场景1: 文档总结
```python
# 长篇文档总结
document = open("long_document.txt").read()
result = processor.process_and_query(
    text=document,
    prompt="请总结这份文档的核心观点",
    model_api_func=openai_api
)
```

### 场景2: 知识库问答
```python
# 基于大量文档的问答
knowledge_base = load_documents()  # 加载多个文档
combined_text = "\n\n".join(knowledge_base)

result = processor.process_and_query(
    text=combined_text,
    prompt="根据以上内容回答：什么是人工智能？",
    model_api_func=openai_api
)
```

### 场景3: 代码分析
```python
# 分析大型代码库
code = open("large_codebase.py").read()

result = processor.process_and_query(
    text=code,
    prompt="请分析这段代码的架构和主要功能",
    model_api_func=openai_api
)
```

### 场景4: 研究论文处理
```python
# 处理长篇学术论文
paper = open("research_paper.pdf").read()  # 需要先提取文本

result = processor.process_and_query(
    text=paper,
    prompt="请提取这篇论文的研究方法、主要发现和结论",
    model_api_func=openai_api
)
```

## 批量处理

```python
# 批量处理多个文档
documents = [
    "文档1内容...",
    "文档2内容...",
    "文档3内容..."
]

results = processor.batch_process(
    texts=documents,
    prompt="请提取关键信息",
    model_api_func=openai_api
)

# 查看结果
for i, result in enumerate(results):
    if result['compression_info']:
        print(f"文档{i+1}: 压缩比 {result['compression_info']['compression_ratio']:.2%}")
```

## 关键参数说明

### context_length（上下文窗口长度）
- GPT-3.5-turbo: 4096 tokens
- GPT-4: 8192 tokens
- GPT-4-32k: 32768 tokens
- Claude-3: 200000 tokens
- LLaMA-2-7B: 4096 tokens

### max_output_tokens（最大输出token数）
- 根据任务需求设置
- 总结任务: 256-512 tokens
- 详细分析: 1024-2048 tokens
- 代码生成: 1024-4096 tokens

### safety_margin（安全余量）
- 默认: 0.1 (10%)
- 作用: 预留token空间，避免超出上下文窗口
- 建议: 0.05-0.15

### enable_compression（是否启用压缩）
- True: 自动压缩超长文本
- False: 不压缩，直接处理（可能超出上下文窗口）

## 性能优化建议

### 1. 选择合适的上下文窗口
```python
# 短文本任务
processor = LongTextProcessor(context_length=4096)

# 长文本任务
processor = LongTextProcessor(context_length=32768)
```

### 2. 调整安全余量
```python
# 保守策略（更安全）
processor = LongTextProcessor(safety_margin=0.15)

# 激进策略（更高效）
processor = LongTextProcessor(safety_margin=0.05)
```

### 3. 使用快速压缩模式
```python
# 在DynamicContextCompressor中设置
compressor = DynamicContextCompressor(
    session_len=4096,
    use_fast_compression=True  # 更快但精度略低
)
```

### 4. 批量处理
```python
# 使用批量处理提高效率
results = processor.batch_process(texts, prompt, api_func)
```

## 错误处理

### API调用失败
```python
try:
    result = client.chat_with_long_context(messages)
except Exception as e:
    print(f"调用失败: {e}")
    # 使用备用方案或重试
```

### 压缩失败
```python
if result['compression_info'] is None:
    print("压缩失败，使用原始文本")
else:
    print(f"压缩成功，压缩比: {result['compression_info']['compression_ratio']:.2%}")
```

## 安装依赖

```bash
# 基础依赖
pip install numpy transformers

# OpenAI API
pip install openai

# Anthropic API
pip install anthropic

# 通用依赖
pip install requests
```

## 运行示例

```bash
# 基础示例
python3 examples/long_text_processing_example.py

# 真实API集成示例
python3 examples/real_llm_integration_example.py
```

## 常见问题

### Q1: 如何判断是否需要压缩？
A: 系统会自动计算总token数（文本+提示词+预留输出），如果超过可用上下文，则自动压缩。

### Q2: 压缩会丢失重要信息吗？
A: 压缩算法会保留关键词、重要段落和核心信息，通常能保留80%以上的关键信息。

### Q3: 压缩速度如何？
A: 通常在0.01-0.1秒之间，取决于文本长度和硬件性能。

### Q4: 可以自定义压缩策略吗？
A: 可以，通过继承DynamicContextCompressor类并重写相关方法。

### Q5: 支持哪些模型？
A: 支持所有提供API接口的大模型，包括OpenAI、Anthropic、本地模型等。

## 总结

使用文本压缩服务处理超长文本的完整流程：

1. **初始化处理器/客户端**
2. **输入超长文本和提示词**
3. **自动检查是否需要压缩**
4. **如果需要，执行智能压缩**
5. **构建符合上下文窗口的请求**
6. **调用大模型API**
7. **获取结果和压缩信息**

整个过程对用户透明，只需几行代码即可完成！
