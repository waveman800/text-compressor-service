"""
压缩服务与大模型上下文窗口关联机制说明
=========================================

## 核心概念

### 1. 上下文窗口长度 (session_len)
- 定义：大模型单次推理能够处理的最大token数量
- 常见值：4096, 8192, 16384, 32768等
- 作用：作为动态压缩的触发阈值

### 2. 预留生成空间 (max_new_tokens)
- 定义：为模型生成新内容预留的token数量
- 作用：确保压缩后的文本有足够空间让模型生成新内容
- 常见值：256, 512, 1024等

## 动态压缩触发机制

### 触发条件
```python
total_tokens = prompt_tokens + text_tokens + max_new_tokens

if total_tokens > session_len:
    # 触发动态压缩
    available_tokens = session_len - prompt_tokens - max_new_tokens
    compressed_text = compress_to_fit(text, available_tokens)
```

### 触发流程
1. **计算当前token数**
   - prompt_tokens: 当前提示词的token数
   - text_tokens: 待压缩文本的token数
   - max_new_tokens: 预留给生成新内容的token数

2. **判断是否需要压缩**
   - 如果 total_tokens <= session_len: 不需要压缩，直接返回原文本
   - 如果 total_tokens > session_len: 触发压缩机制

3. **计算可用token数**
   ```python
   available_tokens = session_len - prompt_tokens - max_new_tokens
   ```
   这是压缩后文本的最大token限制

4. **执行压缩**
   - 根据文本类型选择压缩策略
   - 将文本压缩到 available_tokens 以内

## 压缩策略

### 1. 叙述类型 (narrative)
- 识别：普通文本，无特殊结构
- 策略：按段落分割，基于重要性评分保留关键内容
- 重要性因素：
  - 关键词匹配（70%权重）
  - 简化的TF-IDF分析（30%权重）
  - 位置权重（开头和结尾更重要，25%权重）
  - 长度权重（适中长度更优，15%权重）

### 2. 对话类型 (dialogue)
- 识别：包含引号、对话标记
- 策略：优先保留对话内容，压缩描述性文本
- 优势：保持对话的连贯性和关键信息

### 3. 列表类型 (list)
- 识别：包含编号、项目符号
- 策略：优先保留列表项，压缩描述
- 优势：保持结构化信息的完整性

### 4. 代码类型 (code)
- 识别：包含代码块、函数定义
- 策略：保留代码结构，压缩注释和文档字符串
- 优势：保持代码的可执行性

## 使用示例

### 示例1：单文本压缩
```python
from core.dynamic_compressor import DynamicContextCompressor

# 初始化压缩器
compressor = DynamicContextCompressor(
    session_len=4096,      # 上下文窗口长度
    max_new_tokens=256,     # 预留生成空间
    enable_dynamic_compression=True
)

# 压缩文本
text = "长文本内容..."
compressed_text, was_compressed = compressor.dynamic_compress(
    text=text,
    current_prompt="当前提示词",
    max_new_tokens=256
)
```

### 示例2：聊天历史压缩
```python
# 聊天历史
chat_history = [
    {"role": "user", "content": "用户消息1"},
    {"role": "assistant", "content": "助手回复1"},
    {"role": "user", "content": "用户消息2"},
    # ... 更多消息
]

# 压缩聊天历史
compressed_history, was_compressed = compressor.compress_chat_history(
    chat_history=chat_history,
    current_prompt="当前提示词",
    max_new_tokens=256
)
```

## API接口

### 1. /compress/text
```bash
POST /compress/text
Content-Type: application/json

{
    "text": "要压缩的文本",
    "session_len": 4096,
    "max_new_tokens": 256,
    "current_prompt": "当前提示词（可选）"
}
```

### 2. /compress/chat
```bash
POST /compress/chat
Content-Type: application/json

{
    "chat_history": [
        {"role": "user", "content": "用户消息"},
        {"role": "assistant", "content": "助手回复"}
    ],
    "session_len": 4096,
    "max_new_tokens": 256,
    "current_prompt": "当前提示词（可选）"
}
```

## 性能优化

### 1. 快速压缩模式
```python
compressor = DynamicContextCompressor(
    use_fast_compression=True  # 启用快速压缩
)
```
- 优点：速度更快
- 缺点：压缩质量略低

### 2. 异步处理
```python
# 异步压缩
compressed_text, was_compressed = await compressor.dynamic_compress_async(
    text=text,
    current_prompt=current_prompt,
    max_new_tokens=256
)
```

### 3. 批量处理
```python
# 批量压缩多个文本
results = await compressor.batch_compress(
    texts=[text1, text2, text3],
    current_prompt=current_prompt,
    max_new_tokens=256
)
```

## 压缩质量评估

### 评估指标
```python
metrics = compressor.evaluate_compression(
    original_text=original_text,
    compressed_text=compressed_text,
    processing_time=processing_time
)

# 指标包括：
# - original_tokens: 原始token数
# - compressed_tokens: 压缩后token数
# - compression_ratio: 压缩比（越小越好）
# - keyword_retention: 关键词保留率（越接近1越好）
# - information_preservation: 信息保留率（越接近1越好）
# - processing_time: 处理时间
```

## 与推理框架集成

### 集成步骤
1. **初始化压缩器**
   ```python
   compressor = DynamicContextCompressor(
       session_len=model_config.context_length,
       max_new_tokens=model_config.max_output_tokens
   )
   ```

2. **在推理前压缩**
   ```python
   def inference_with_compression(prompt, context):
       # 压缩上下文
       compressed_context, _ = compressor.dynamic_compress(
           text=context,
           current_prompt=prompt,
           max_new_tokens=model_config.max_output_tokens
       )
       
       # 使用压缩后的上下文进行推理
       return model.generate(prompt + compressed_context)
   ```

3. **处理聊天历史**
   ```python
   def chat_with_compression(messages, new_message):
       # 压缩历史消息
       compressed_history, _ = compressor.compress_chat_history(
           chat_history=messages,
           current_prompt=new_message,
           max_new_tokens=model_config.max_output_tokens
       )
       
       # 使用压缩后的历史进行推理
       return model.generate(compressed_history + [new_message])
   ```

## 注意事项

1. **session_len设置**
   - 应该根据模型的实际上下文窗口长度设置
   - 不同模型的上下文窗口长度不同
   - 建议留出10-20%的安全余量

2. **max_new_tokens设置**
   - 应该根据预期的输出长度设置
   - 设置过大会浪费压缩空间
   - 设置过小可能导致生成内容被截断

3. **压缩质量**
   - 压缩会损失部分信息
   - 对于关键信息，建议手动调整或使用摘要
   - 可以通过调整关键词列表来优化压缩质量

4. **性能考虑**
   - 压缩操作会增加推理延迟
   - 对于实时性要求高的场景，建议使用快速压缩模式
   - 可以预压缩常用文本以减少实时计算

## 常见问题

### Q1: 如何确定session_len的值？
A: 查看模型配置文件或文档，不同模型的上下文窗口长度不同：
- GPT-3.5: 4096
- GPT-4: 8192/32768
- Claude: 100000
- LLaMA: 2048/4096

### Q2: 压缩后的文本质量如何保证？
A: 通过以下方式保证质量：
- 基于重要性评分保留关键内容
- 根据文本类型选择最优策略
- 保留关键词和关键信息
- 可以通过评估指标监控压缩质量

### Q3: 如何优化压缩速度？
A: 优化方法：
- 启用快速压缩模式
- 使用异步处理
- 预计算token数
- 批量处理多个文本

### Q4: 压缩是否会丢失重要信息？
A: 可能会丢失部分信息，但通过以下方式减少损失：
- 基于重要性评分
- 保留关键词
- 保留开头和结尾内容
- 可以手动调整压缩结果
